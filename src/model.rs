use crate::archive::CqGather;
use crate::math::{axpy, dot, norm, quant, rms, rope, sigmoid, sinkhorn, softmax};
use crate::{Archive, CancellationToken, EmbeddedRuntime, Error, Geometry, Result, Tokenizer};
use a3s_power::inference::{ExecutionDigest, ExecutionReceipt, ModelIdentity};
use candle_core::Tensor;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};

static PROCESS_STOP: AtomicBool = AtomicBool::new(false);

/// Process-wide stop set by the CLI signal handler. The handler may only
/// store this flag; inference polls it from normal code and then cancels the
/// caller token. A second OS signal is handled by the CLI, not here.
pub fn request_process_stop() {
    PROCESS_STOP.store(true, Ordering::SeqCst);
}

pub fn process_stop_requested() -> bool {
    PROCESS_STOP.load(Ordering::SeqCst)
}

/// Reference CPU implementation. Power supplies admission and the tensor device.
pub struct Model {
    pub(crate) tokenizer: Tokenizer,
    geometry: Geometry,
    sha256: String,
    runtime: EmbeddedRuntime,
    weights: Vec<Tensor>,
    /// Engram gather tables that stay packed. `None` means the tensor is in `weights`.
    gather: Vec<Option<CqGather>>,
    /// Complete projections stored once as contiguous `[in, out]`.
    /// Logical archive shape remains `[out, in]`.
    gemm: Vec<bool>,
    codebook: Vec<f32>,
    mhc: usize,
    engrams: usize,
    final_norm: usize,
    p1: Vec<usize>,
    p2: Vec<usize>,
    heads: Vec<(u8, usize)>,
}

#[derive(Default)]
struct LayerState {
    q: VecDeque<Vec<f32>>,
    k: VecDeque<Vec<f32>>,
    v: VecDeque<Vec<f32>>,
    keys: VecDeque<Vec<f32>>,
    values: VecDeque<Vec<f32>>,
}

struct State {
    tokens: Vec<u32>,
    layers: Vec<LayerState>,
    engram_values: Vec<VecDeque<Vec<f32>>>,
    cells: Vec<Vec<f32>>,
    collect_cells: bool,
    capture_layers: bool,
    layer_means: Vec<Vec<f32>>,
    quantized_hidden: Vec<f32>,
}

struct ProjectedAttention<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    gates: &'a [f32],
}

struct AttentionLog<'a> {
    queries: &'a [f32],
    keys: &'a [f32],
    values: &'a [f32],
    spans: &'a [(usize, usize)],
}

struct TokenCarry {
    u: Vec<f32>,
    x: Vec<f32>,
    post: Vec<f32>,
    res: Vec<f32>,
}

/// Last-token lane means and the activation-quantized final hidden state.
#[derive(Debug)]
pub struct LayerReadout {
    pub layers: Vec<Vec<f32>>,
    pub quantized_hidden: Vec<f32>,
}

#[derive(Debug, serde::Serialize)]
pub struct Generation {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub input_tokens: usize,
    pub finished: bool,
    /// Mean probability of the selected tokens; not a calibrated correctness score.
    pub mean_token_probability: f32,
    /// Raw sigmoid of the learned head. Domain calibration is a host obligation.
    pub head_score: Option<f32>,
    pub receipt: ExecutionReceipt,
}

impl Model {
    pub fn load(archive: Archive, runtime: EmbeddedRuntime) -> Result<Self> {
        let g = &archive.geometry;
        if runtime.device().kind() != a3s_power::inference::RuntimeDeviceKind::Cpu {
            return Err(Error::Request(
                "reference implementation requires the Power CPU device".into(),
            ));
        }
        if g.kv_bits != 8 {
            return Err(Error::Archive("only A8 KV archives are supported".into()));
        }
        let stride = if g.conv_taps > 0 { 27 } else { 24 };
        let mhc = 1 + stride * g.layers;
        let engrams = mhc + 11;
        let final_norm = engrams + 4 * g.engram_sites.len();
        if archive.records.len() < final_norm + 2 {
            return Err(Error::Archive("missing model tensors".into()));
        }
        let gather_tables: Vec<usize> = (0..g.engram_sites.len())
            .map(|site| engrams + 4 * site)
            .collect();
        let mut bytes = 0u64;
        for (index, rec) in archive.records[..archive.records.len() - 1]
            .iter()
            .enumerate()
        {
            let packed = rec.dtype == 3 && gather_tables.contains(&index);
            let n = if packed {
                rec.bytes as u64
            } else {
                (runtime
                    .limits()
                    .checked_elements(&rec.shape, "model tensor")? as u64)
                    .checked_mul(4)
                    .ok_or_else(|| Error::Archive("resident size overflow".into()))?
            };
            bytes = bytes
                .checked_add(n)
                .ok_or_else(|| Error::Archive("resident size overflow".into()))?;
        }
        if bytes > runtime.limits().max_resident_weight_bytes {
            return Err(Error::Archive(
                "resident weights exceed Power residency limit".into(),
            ));
        }
        let tokenizer = archive.tokenizer()?;
        // Engram tables are row gathers, so they stay CQ. Complete projections
        // expand once into contiguous [in, out]. A step then uses the contiguous
        // GEMV. Sliced mixes stay [out, in] because a step narrows output rows.
        let mut gemm = vec![false; archive.records.len() - 1];
        for index in full_projection_indices(g) {
            if index >= gemm.len() || gemm[index] {
                return Err(Error::Archive("projection index is not a tensor".into()));
            }
            gemm[index] = true;
        }
        let mut weights = Vec::new();
        let mut gather = Vec::new();
        let device = runtime.device().tensor_device();
        let tensor_count = archive.records.len() - 1;
        let specs: Vec<(u8, Vec<usize>)> = archive.records[..tensor_count]
            .iter()
            .map(|record| (record.dtype, record.shape.clone()))
            .collect();
        for (i, (dtype, shape)) in specs.into_iter().enumerate() {
            let is_gather = dtype == 3 && gather_tables.contains(&i);
            if gemm[i] && is_gather {
                return Err(Error::Archive("projection overlaps a gather table".into()));
            }
            if is_gather {
                gather.push(Some(archive.cq_gather(i)?));
                weights.push(Tensor::from_vec(vec![0.0], &[1], device)?);
            } else if gemm[i] {
                if shape.len() != 2 {
                    return Err(Error::Archive("projection rank mismatch".into()));
                }
                let mut values = archive.values(i)?;
                transpose_out_in_inplace(&mut values, shape[0], shape[1])?;
                weights.push(Tensor::from_vec(values, &[shape[1], shape[0]], device)?);
                gather.push(None);
            } else {
                gather.push(None);
                weights.push(Tensor::from_vec(
                    archive.values(i)?,
                    shape.as_slice(),
                    device,
                )?);
            }
        }
        let codebook = archive.codebook().to_vec();
        let permutation = |index: usize| -> Result<Vec<usize>> {
            let v = weights[index].flatten_all()?.to_vec1::<f32>()?;
            let mut seen = vec![false; g.hadamard];
            if v.len() != g.hadamard {
                return Err(Error::Archive("permutation shape mismatch".into()));
            }
            for &f in &v {
                if f < 0.0 || f.fract() != 0.0 || f as usize >= seen.len() || seen[f as usize] {
                    return Err(Error::Archive("invalid Hadamard permutation".into()));
                }
                seen[f as usize] = true;
            }
            Ok(v.into_iter().map(|v| v as usize).collect())
        };
        let p1 = permutation(mhc + 9)?;
        let p2 = permutation(mhc + 10)?;
        let mut heads = Vec::new();
        if weights.len() > final_norm + 1 {
            let codes = weights[final_norm + 1].flatten_all()?.to_vec1::<f32>()?;
            let mut next = final_norm + 2;
            let mut previous = 0;
            for code in codes {
                if ![1.0, 2.0, 3.0].contains(&code) || code as u8 <= previous {
                    return Err(Error::Archive("invalid probe head manifest".into()));
                }
                let code = code as u8;
                heads.push((code, next));
                next += if code == 3 { 7 } else { 6 };
                previous = code;
            }
            if next != weights.len() {
                return Err(Error::Archive("probe tensor count mismatch".into()));
            }
        }
        let mut model = Self {
            tokenizer,
            geometry: archive.geometry,
            sha256: archive.sha256,
            runtime,
            weights,
            gather,
            gemm,
            codebook,
            mhc,
            engrams,
            final_norm,
            p1,
            p2,
            heads,
        };
        model.validate_shapes()?;
        // Probe projections are addressed by the manifest, which is known only
        // after the tensors exist. Orient them after the archive-shape check.
        for head in 0..model.heads.len() {
            let index = model.heads[head].1 + 4;
            if model.gemm.get(index).copied().unwrap_or(true) {
                return Err(Error::Archive("probe projection index collision".into()));
            }
            let oriented = orient_tensor(
                model
                    .weights
                    .get(index)
                    .ok_or_else(|| Error::Archive("missing probe projection".into()))?,
            )?;
            model.weights[index] = oriented;
            model.gemm[index] = true;
        }
        Ok(model)
    }

    pub fn runtime(&self) -> &EmbeddedRuntime {
        &self.runtime
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    fn shape(&self, i: usize, expected: &[usize]) -> Result<()> {
        let matches = if let Some(Some(table)) = self.gather.get(i) {
            table.shape.as_slice() == expected
        } else if self.gemm.get(i).copied().unwrap_or(false) {
            let dims = self.weights.get(i).map(|t| t.dims()).unwrap_or(&[]);
            expected.len() == 2
                && dims.len() == 2
                && dims[0] == expected[1]
                && dims[1] == expected[0]
        } else {
            self.weights.get(i).is_some_and(|t| t.dims() == expected)
        };
        if !matches {
            return Err(Error::Archive(format!(
                "tensor {i} expected shape {expected:?}"
            )));
        }
        Ok(())
    }

    fn validate_shapes(&self) -> Result<()> {
        let g = &self.geometry;
        let d = g.width;
        let n = g.lanes;
        let l = g.layers;
        self.shape(0, &[g.vocab, d])?;
        let ba = 1 << (g.hadamard.ilog2() / 2);
        let bb = g.hadamard / ba;
        for layer in 0..l {
            let b = self.base(layer);
            let shift = if g.conv_taps > 0 { 3 } else { 0 };
            self.shape(b, &[d])?;
            self.shape(b + 1, &[g.heads * g.qk_dim, d])?;
            self.shape(b + 2, &[g.kv_heads * g.qk_dim, d])?;
            self.shape(b + 3, &[g.kv_heads * g.v_dim, d])?;
            if shift > 0 {
                self.shape(b + 4, &[g.conv_taps, g.heads * g.qk_dim])?;
                self.shape(b + 5, &[g.conv_taps, g.kv_heads * g.qk_dim])?;
                self.shape(b + 6, &[g.conv_taps, g.kv_heads * g.v_dim])?;
            }
            let o = b + shift;
            self.shape(o + 4, &[g.qk_dim])?;
            self.shape(o + 5, &[g.qk_dim])?;
            self.shape(o + 6, &[g.heads * g.v_dim, d])?;
            self.shape(o + 7, &[d, g.heads * g.v_dim])?;
            self.shape(o + 8, &[d])?;
            self.shape(o + 9, &[1])?;
            self.shape(o + 10, &[d])?;
            for j in 11..16 {
                self.shape(o + j, &[g.hadamard])?;
            }
            for j in [16, 18, 20] {
                self.shape(o + j, &[ba, ba])?;
                self.shape(o + j + 1, &[bb, bb])?;
            }
            self.shape(o + 22, &[d, 8])?;
            self.shape(o + 23, &[8, g.hadamard])?;
        }
        for j in 0..3 {
            self.shape(self.mhc + j, &[l])?;
        }
        for j in [3, 4] {
            self.shape(self.mhc + j, &[l, n])?;
        }
        self.shape(self.mhc + 5, &[l, n, n])?;
        for j in [6, 7] {
            self.shape(self.mhc + j, &[l * n, n * d])?;
        }
        self.shape(self.mhc + 8, &[l * n * n, n * d])?;
        for site in 0..g.engram_sites.len() {
            let b = self.engrams + 4 * site;
            self.shape(b, &[g.engram_tables * g.engram_slots, g.engram_dim])?;
            for j in [1, 2] {
                self.shape(b + j, &[d, g.engram_tables * g.engram_dim])?;
            }
            self.shape(b + 3, &[g.engram_taps, d])?;
        }
        self.shape(self.final_norm, &[d])?;
        for &(code, b) in &self.heads {
            let gain = self
                .weights
                .get(b + 1)
                .ok_or_else(|| Error::Archive("missing probe gain".into()))?;
            let (rows, k) = gain.dims2()?;
            let q = self
                .weights
                .get(b + 2)
                .ok_or_else(|| Error::Archive("missing probe queries".into()))?
                .dims2()?
                .0;
            if rows != l + 1 || k == 0 || q == 0 {
                return Err(Error::Archive("invalid probe geometry".into()));
            }
            self.shape(b, &[rows * k, d])?;
            self.shape(b + 2, &[q, d])?;
            self.shape(b + 3, &[q, rows, k])?;
            let out = self
                .weights
                .get(b + 4)
                .ok_or_else(|| Error::Archive("missing probe projection".into()))?
                .dims2()?
                .0;
            if (code == 2 && out != 1) || (code == 3 && out != 3) {
                return Err(Error::Archive("invalid probe output width".into()));
            }
            self.shape(b + 4, &[out, q * d])?;
            self.shape(b + 5, &[out])?;
            if code == 3 {
                self.shape(b + 6, &[3])?;
            }
        }
        Ok(())
    }

    fn base(&self, layer: usize) -> usize {
        1 + layer * if self.geometry.conv_taps > 0 { 27 } else { 24 }
    }
    fn flat(&self, i: usize) -> Result<Vec<f32>> {
        Ok(self.weights[i].flatten_all()?.to_vec1::<f32>()?)
    }
    fn row(&self, i: usize, row: usize) -> Result<Vec<f32>> {
        if let Some(Some(table)) = self.gather.get(i) {
            return table.row(row, &self.codebook);
        }
        if self.gemm.get(i).copied().unwrap_or(false) {
            // Archive row `r` is column `r` of the contiguous `[in, out]` matrix.
            return Ok(self.weights[i].get_on_dim(1, row)?.to_vec1::<f32>()?);
        }
        Ok(self.weights[i].get(row)?.flatten_all()?.to_vec1::<f32>()?)
    }
    /// `w` is contiguous `[in, out]`.
    fn gemv(&self, w: &Tensor, x: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_slice(x, (1, x.len()), self.runtime.device().tensor_device())?;
        Ok(x.matmul(w)?.flatten_all()?.to_vec1::<f32>()?)
    }
    /// `rows` is row-major `[count, cols]`. One row stays on `gemv`.
    fn project_matrix(&self, w: &Tensor, rows: &[f32], cols: usize) -> Result<Vec<f32>> {
        if cols == 0 || !rows.len().is_multiple_of(cols) {
            return Err(Error::Request("projection width mismatch".into()));
        }
        let count = rows.len() / cols;
        if count == 0 {
            return Ok(Vec::new());
        }
        if count == 1 {
            return self.gemv(w, rows);
        }
        let dims = w.dims();
        if dims.len() != 2 || dims[0] != cols {
            return Err(Error::Request("projection width mismatch".into()));
        }
        let out = dims[1];
        let x = Tensor::from_slice(rows, (count, cols), self.runtime.device().tensor_device())?;
        let y = x.matmul(w)?.flatten_all()?.to_vec1::<f32>()?;
        if y.len() != count * out {
            return Err(Error::Request("projection width mismatch".into()));
        }
        Ok(y)
    }
    /// One GEMM per projection for a long prompt. A 768-wide product is faster
    /// from 16 tokens, but a 59-token prompt moved a confidence head by more
    /// than 0.01, so prompts below 128 tokens stay on the token loop. Narrow
    /// graphs always stay there, and each of those tokens remains a
    /// cancellation point.
    fn batched_prefill(&self, tokens: usize) -> bool {
        tokens >= 128 && self.geometry.width >= 64
    }
    /// `w` is archive layout `[out, in]`, usually a narrow output-row slice.
    fn linear_tensor(&self, w: &Tensor, x: &[f32]) -> Result<Vec<f32>> {
        self.gemv(&w.t()?, x)
    }
    fn linear(&self, i: usize, x: &[f32]) -> Result<Vec<f32>> {
        if self.gemm.get(i).copied().unwrap_or(false) {
            self.gemv(&self.weights[i], x)
        } else {
            self.linear_tensor(&self.weights[i], x)
        }
    }
    fn affine_input(&self, i: usize, x: &[f32]) -> Result<Vec<f32>> {
        // These tensors are already stored as `[in, out]`.
        self.gemv(&self.weights[i], x)
    }
    fn state(&self, collect_cells: bool) -> State {
        State {
            tokens: Vec::new(),
            layers: (0..self.geometry.layers)
                .map(|_| LayerState::default())
                .collect(),
            engram_values: vec![VecDeque::new(); self.geometry.engram_sites.len()],
            cells: Vec::new(),
            collect_cells,
            capture_layers: false,
            layer_means: Vec::new(),
            quantized_hidden: Vec::new(),
        }
    }

    fn capturing_state(&self) -> State {
        let mut state = self.state(false);
        state.capture_layers = true;
        state
    }

    /// Lane-mean stream after every layer, then the deployed A8 final hidden.
    /// The caller token and the process stop flag are both observed.
    pub fn layer_readout(
        &self,
        tokens: &[u32],
        cancel: &CancellationToken,
    ) -> Result<LayerReadout> {
        let _permit = self.runtime.begin(cancel)?;
        self.check_request(tokens, 0, false, 0)?;
        let mut state = self.capturing_state();
        self.prefill(tokens, &mut state, cancel, false)?;
        check_cancel(cancel)?;
        let width = self.geometry.width;
        if state.layer_means.len() != self.geometry.layers
            || state.quantized_hidden.len() != width
            || state.layer_means.iter().any(|layer| layer.len() != width)
            || state
                .quantized_hidden
                .iter()
                .any(|value| !value.is_finite())
            || state
                .layer_means
                .iter()
                .any(|layer| layer.iter().any(|value| !value.is_finite()))
        {
            return Err(Error::Request("non-finite layer readout".into()));
        }
        Ok(LayerReadout {
            layers: state.layer_means,
            quantized_hidden: state.quantized_hidden,
        })
    }

    fn check_request(
        &self,
        tokens: &[u32],
        additional: usize,
        collect_cells: bool,
        reserved_bytes: u64,
    ) -> Result<()> {
        let total = tokens
            .len()
            .checked_add(additional)
            .ok_or_else(|| Error::Request("context size overflow".into()))?;
        let g = &self.geometry;
        let lim = self.runtime.limits();
        if tokens.is_empty()
            || total > g.context.min(lim.max_context_tokens)
            || additional > lim.max_generated_tokens
            || tokens.iter().any(|&v| v as usize >= g.vocab)
        {
            return Err(Error::Request("token or context limit exceeded".into()));
        }
        let mut bytes = reserved_bytes;
        let vector = std::mem::size_of::<Vec<f32>>();
        // Allow Vec/VecDeque growth and overlapping input/output temporaries.
        add_bytes(&mut bytes, &[2, total, 4])?;
        add_bytes(&mut bytes, &[2, tokens.len(), 4])?;
        add_bytes(
            &mut bytes,
            &[2, g.layers + 4, std::mem::size_of::<LayerState>()],
        )?;
        for layer in 0..g.layers {
            let conv = total.min(g.conv_taps.max(1));
            let window = self.attention_window(layer, total);
            for (count, width) in [
                (conv, g.heads * g.qk_dim),
                (conv, g.kv_heads * g.qk_dim),
                (conv, g.kv_heads * g.v_dim),
                (window, g.kv_heads * g.qk_dim),
                (window, g.kv_heads * g.v_dim),
            ] {
                add_bytes(&mut bytes, &[2, count, width, 4])?;
                add_bytes(&mut bytes, &[2, count + 4, vector])?;
            }
        }
        if !g.engram_sites.is_empty() {
            let history = total.min(1 + (g.engram_taps - 1) * g.engram_dilation);
            add_bytes(&mut bytes, &[2, g.engram_sites.len(), history, g.width, 4])?;
            add_bytes(&mut bytes, &[2, g.engram_sites.len(), history + 4, vector])?;
        }
        if collect_cells {
            lim.checked_elements(&[total, g.layers + 1, g.width], "head cells")?;
            add_bytes(&mut bytes, &[2, total, g.layers + 1, g.width, 4])?;
            add_bytes(&mut bytes, &[2, total + 4, vector])?;
        }
        // Reference forward vectors and cloned norm/convolution/Monarch weights.
        for elements in [
            g.width * (8 * g.lanes + 32),
            32 * g.hadamard,
            8 * g.lanes * g.lanes,
            8 * g.heads * (g.qk_dim + g.v_dim),
            8 * g.kv_heads * (g.qk_dim + g.v_dim),
            4 * g.out_vocab,
            4 * g.engram_tables * g.engram_dim,
            3 * g.layers,
            g.conv_taps * ((g.heads + g.kv_heads) * g.qk_dim + g.kv_heads * g.v_dim),
            g.engram_taps * g.width,
        ] {
            add_bytes(&mut bytes, &[elements, 4])?;
        }
        if collect_cells {
            let mut largest = 0u64;
            for &(_, b) in &self.heads {
                let rows = g.layers + 1;
                let k = self.weights[b + 1].dims()[1];
                let q = self.weights[b + 2].dims()[0];
                let mut head = 0;
                add_bytes(&mut head, &[2, rows, k, g.width, 4])?;
                add_bytes(&mut head, &[2, rows * k + 4, vector])?;
                add_bytes(&mut head, &[2, rows, k, 4])?;
                add_bytes(&mut head, &[2, q, rows, k, 4])?;
                add_bytes(&mut head, &[2, q + 4, g.width, 4])?;
                add_bytes(&mut head, &[4, self.weights[b + 4].dims()[0], 4])?;
                add_bytes(&mut head, &[2, total + rows * k + 8, 4])?;
                largest = largest.max(head);
            }
            bytes = bytes
                .checked_add(largest)
                .ok_or_else(|| Error::Request("state size overflow".into()))?;
        }
        add_bytes(
            &mut bytes,
            &[16, additional, self.tokenizer.max_token_bytes()],
        )?;
        lim.checked_state_bytes(bytes, "Needle reference state")?;
        Ok(())
    }

    fn attention_window(&self, layer: usize, available: usize) -> usize {
        let g = &self.geometry;
        let mut window = available;
        if g.global_mask & (1 << layer) == 0 && g.sliding_window > 0 {
            window = window.min(g.sliding_window);
        }
        if g.kv_window > 0 {
            window = window.min(g.kv_window);
        }
        window
    }

    pub fn logits(&self, tokens: &[u32], cancel: &CancellationToken) -> Result<Vec<f32>> {
        let _permit = self.runtime.begin(cancel)?;
        self.check_request(tokens, 0, false, 0)?;
        let mut state = self.state(false);
        let logits = self.prefill(tokens, &mut state, cancel, true)?;
        check_cancel(cancel)?;
        Ok(logits)
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &CancellationToken,
    ) -> Result<Generation> {
        let _permit = self.runtime.begin(cancel)?;
        self.generate_admitted(prompt, max_tokens, cancel, None, 0)
    }

    pub(crate) fn generate_admitted(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &CancellationToken,
        schema: Option<&crate::structured::Schema>,
        reserved_bytes: u64,
    ) -> Result<Generation> {
        if prompt.len() > self.runtime.limits().max_input_bytes || max_tokens == 0 {
            return Err(Error::Request("invalid prompt or generation bound".into()));
        }
        let mut token_limits = self.runtime.limits().clone();
        token_limits.max_state_bytes = token_limits
            .max_state_bytes
            .checked_sub(reserved_bytes)
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Request("tool schema exhausted the state budget".into()))?;
        let mut input = vec![self.tokenizer.bos];
        input.extend(
            self.tokenizer
                .encode_bounded(prompt, &token_limits, cancel)?,
        );
        let collect_cells = self.heads.iter().any(|(code, _)| *code == 2);
        self.check_request(&input, max_tokens, collect_cells, reserved_bytes)?;
        let mut state = self.state(collect_cells);
        let mut logits = self.prefill(&input, &mut state, cancel, true)?;
        let mut output = Vec::new();
        let mut sum = 0.0;
        let mut finished = false;
        let mut body: Option<Vec<u8>> = None;
        for _ in 0..max_tokens {
            check_cancel(cancel)?;
            if let (Some(schema), Some(bytes)) = (schema, body.as_ref()) {
                checkpoint(cancel, "grammar")?;
                for (id, logit) in logits.iter_mut().enumerate() {
                    if id.is_multiple_of(64) {
                        check_cancel(cancel)?;
                    }
                    let allowed = if Some(id as u32) == self.tokenizer.token_id("</tool_call>") {
                        schema.accepts(bytes, true)
                    } else {
                        let piece = self.tokenizer.token_bytes(id as u32)?;
                        let mut candidate = bytes.clone();
                        candidate.extend(&piece);
                        !piece.is_empty() && schema.accepts(&candidate, false)
                    };
                    if !allowed {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                if logits.iter().all(|v| !v.is_finite()) {
                    return Err(Error::Request(
                        "no token satisfies the output grammar".into(),
                    ));
                }
            }
            let id = logits
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(i, _)| i)
                .ok_or_else(|| Error::Request("empty logits".into()))?;
            softmax(&mut logits);
            sum += logits[id];
            output.push(id as u32);
            if id as u32 == self.tokenizer.eos
                || Some(id as u32) == self.tokenizer.token_id("</tool_call>")
            {
                finished = true;
                break;
            }
            if body.is_none() && Some(id as u32) == self.tokenizer.token_id("<tool_call>") {
                body = Some(Vec::new());
            } else if let Some(bytes) = &mut body {
                bytes.extend(self.tokenizer.token_bytes(id as u32)?);
            }
            logits = self.step(id as u32, &mut state, cancel, StepLogits::Return)?;
        }
        // Include the final selected token in the post-hoc head's context.
        // That pass updates the cache. Its logits are not sampled.
        if finished {
            if let Some(&last) = output.last() {
                self.step(last, &mut state, cancel, StepLogits::Omit)?;
            }
        }
        let head_score = self
            .heads
            .iter()
            .find(|(code, _)| *code == 2)
            .map(|&(_, b)| {
                checkpoint(cancel, "head")?;
                self.probe(&state, b, cancel).and_then(|pool| {
                    let score = self.linear(b + 4, &pool)?[0] + self.flat(b + 5)?[0];
                    if !score.is_finite() {
                        return Err(Error::Request("non-finite confidence head".into()));
                    }
                    Ok(sigmoid(score))
                })
            })
            .transpose()?;
        check_cancel(cancel)?;
        let receipt = self.runtime.receipt(
            ModelIdentity::new("needle3", env!("CARGO_PKG_VERSION"), &self.sha256),
            ExecutionDigest::token_ids(&input),
            ExecutionDigest::token_ids(&output),
        );
        let text = self.tokenizer.decode(&output)?;
        checkpoint(cancel, "before-return")?;
        Ok(Generation {
            text,
            mean_token_probability: sum / output.len() as f32,
            token_ids: output,
            input_tokens: input.len(),
            finished,
            head_score,
            receipt,
        })
    }

    pub fn embed(&self, text: &str, cancel: &CancellationToken) -> Result<Vec<f32>> {
        let _permit = self.runtime.begin(cancel)?;
        if text.len() > self.runtime.limits().max_input_bytes {
            return Err(Error::Request("input byte limit exceeded".into()));
        }
        let mut ids = vec![self.tokenizer.bos];
        ids.extend(
            self.tokenizer
                .encode_bounded(text, self.runtime.limits(), cancel)?,
        );
        self.check_request(&ids, 0, true, 0)?;
        let mut state = self.state(true);
        self.prefill(&ids, &mut state, cancel, false)?;
        let &(code, b) = self
            .heads
            .iter()
            .find(|(code, _)| *code == 1)
            .or_else(|| self.heads.iter().find(|(code, _)| *code == 2))
            .ok_or_else(|| Error::Request("archive has no embedding or confidence probe".into()))?;
        let mut v = self.probe(&state, b, cancel)?;
        if code == 1 {
            v = self.linear(b + 4, &v)?;
        }
        let scale = (dot(&v, &v) + 1e-12).sqrt();
        if !scale.is_finite() || v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Request("non-finite embedding head".into()));
        }
        for x in &mut v {
            *x /= scale;
        }
        check_cancel(cancel)?;
        Ok(v)
    }

    fn probe(&self, state: &State, b: usize, cancel: &CancellationToken) -> Result<Vec<f32>> {
        let d = self.geometry.width;
        let rows = self.geometry.layers + 1;
        let k = self.weights[b + 1].dims()[1];
        let q = self.weights[b + 2].dims()[0];
        let gain = self.flat(b + 1)?;
        let bias = self.flat(b + 3)?;
        let mut pooled = Vec::new();
        for row in 0..rows {
            for p in 0..k {
                check_cancel(cancel)?;
                let probe = self.row(b, row * k + p)?;
                let mut scores: Vec<f32> = state
                    .cells
                    .iter()
                    .zip(&state.tokens)
                    .enumerate()
                    .map(|(i, (t, &id))| {
                        if i.is_multiple_of(64) {
                            check_cancel(cancel)?;
                        }
                        Ok(if id == self.tokenizer.pad {
                            f32::NEG_INFINITY
                        } else {
                            dot(&t[row * d..(row + 1) * d], &probe) / (d as f32).sqrt()
                        })
                    })
                    .collect::<Result<_>>()?;
                softmax(&mut scores);
                let mut r = vec![0.0; d];
                for (at, (t, &p)) in state.cells.iter().zip(&scores).enumerate() {
                    if at.is_multiple_of(64) {
                        check_cancel(cancel)?;
                    }
                    for i in 0..d {
                        r[i] += p * t[row * d + i];
                    }
                }
                pooled.push(
                    rms(&r)
                        .iter()
                        .map(|v| v * gain[row * k + p])
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut out = Vec::new();
        for query in 0..q {
            check_cancel(cancel)?;
            let w = self.row(b + 2, query)?;
            let mut scores: Vec<f32> = pooled
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    if i.is_multiple_of(64) {
                        check_cancel(cancel)?;
                    }
                    Ok(dot(r, &w) / (d as f32).sqrt() + bias[query * rows * k + i])
                })
                .collect::<Result<_>>()?;
            softmax(&mut scores);
            let mut r = vec![0.0; d];
            for (at, (v, p)) in pooled.iter().zip(scores).enumerate() {
                if at.is_multiple_of(64) {
                    check_cancel(cancel)?;
                }
                for i in 0..d {
                    r[i] += p * v[i];
                }
            }
            out.extend(r);
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error::Request("non-finite probe output".into()));
        }
        Ok(out)
    }

    fn tapped(
        &self,
        history: &VecDeque<Vec<f32>>,
        weight: usize,
        taps: usize,
        dilation: usize,
    ) -> Result<Vec<f32>> {
        let Some(last) = history.back() else {
            return Err(Error::Request("empty convolution state".into()));
        };
        let mut out = vec![0.0; last.len()];
        let w = self.flat(weight)?;
        for j in 0..taps {
            if let Some(pos) = history.len().checked_sub(1 + j * dilation) {
                for i in 0..out.len() {
                    out[i] += history[pos][i] * w[j * out.len() + i];
                }
            }
        }
        Ok(out)
    }

    fn engram_embedding(&self, site: usize, tokens: &[u32], pos: usize) -> Result<Vec<f32>> {
        if pos >= tokens.len() {
            return Err(Error::Request(
                "engram position is outside the token state".into(),
            ));
        }
        let g = &self.geometry;
        let heads = g.engram_tables / g.engram_orders.len();
        let stride = if g.engram_seed_heads == 0 {
            heads
        } else {
            g.engram_seed_heads
        };
        let mut e = Vec::new();
        let base = self.engrams + 4 * site;
        for (oi, &order) in g.engram_orders.iter().enumerate() {
            for head in 0..heads {
                if pos + 1 < order {
                    e.extend(vec![0.0; g.engram_dim]);
                    continue;
                }
                let mut acc = 0x9e3779b9u32.wrapping_mul((oi * stride + head + 1) as u32);
                for j in 0..order {
                    let id = tokens[pos - j];
                    acc = (acc ^ id).wrapping_mul(0x01000193);
                }
                acc ^= acc >> 15;
                e.extend(self.row(
                    base,
                    (oi * heads + head) * g.engram_slots + acc as usize % g.engram_slots,
                )?);
            }
        }
        quant(&mut e);
        Ok(e)
    }

    fn push_engram(
        &self,
        site: usize,
        state: &mut State,
        k: Vec<f32>,
        v: Vec<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let g = &self.geometry;
        let base = self.engrams + 4 * site;
        push_recent(
            &mut state.engram_values[site],
            v,
            1 + (g.engram_taps - 1) * g.engram_dilation,
        );
        Ok((
            k,
            self.tapped(
                &state.engram_values[site],
                base + 3,
                g.engram_taps,
                g.engram_dilation,
            )?,
        ))
    }

    fn engram(&self, site: usize, state: &mut State, pos: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let e = self.engram_embedding(site, &state.tokens, pos)?;
        let base = self.engrams + 4 * site;
        let k = self.linear(base + 1, &e)?;
        let v = self.linear(base + 2, &e)?;
        self.push_engram(site, state, k, v)
    }

    fn attention(
        &self,
        x: &[f32],
        layer: usize,
        pos: usize,
        state: &mut LayerState,
    ) -> Result<Vec<f32>> {
        let g = &self.geometry;
        let b = self.base(layer);
        let o = b + if g.conv_taps > 0 { 3 } else { 0 };
        let mut x = x.to_vec();
        quant(&mut x);
        let q = self.linear(b + 1, &x)?;
        let k = self.linear(b + 2, &x)?;
        let v = self.linear(b + 3, &x)?;
        let gates = self.linear(o + 6, &x)?;
        let mixed = self.mix_attention(
            layer,
            pos,
            state,
            &ProjectedAttention {
                q: &q,
                k: &k,
                v: &v,
                gates: &gates,
            },
        )?;
        self.linear(o + 7, &mixed)
    }

    /// Convolution, norms, RoPE, and quantization shared by both attention paths.
    fn staged_qkv(
        &self,
        layer: usize,
        pos: usize,
        state: &mut LayerState,
        projected: &ProjectedAttention<'_>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let g = &self.geometry;
        let b = self.base(layer);
        let o = b + if g.conv_taps > 0 { 3 } else { 0 };
        let history = g.conv_taps.max(1);
        push_recent(&mut state.q, projected.q.to_vec(), history);
        push_recent(&mut state.k, projected.k.to_vec(), history);
        push_recent(&mut state.v, projected.v.to_vec(), history);
        let (q, k, v) = if g.conv_taps > 0 {
            (
                self.tapped(&state.q, b + 4, g.conv_taps, 1)?,
                self.tapped(&state.k, b + 5, g.conv_taps, 1)?,
                self.tapped(&state.v, b + 6, g.conv_taps, 1)?,
            )
        } else {
            (
                state
                    .q
                    .back()
                    .ok_or_else(|| Error::Request("empty convolution state".into()))?
                    .clone(),
                state
                    .k
                    .back()
                    .ok_or_else(|| Error::Request("empty convolution state".into()))?
                    .clone(),
                state
                    .v
                    .back()
                    .ok_or_else(|| Error::Request("empty convolution state".into()))?
                    .clone(),
            )
        };
        let qscale = self.flat(o + 4)?;
        let kscale = self.flat(o + 5)?;
        let mut q: Vec<f32> = q
            .chunks_exact(g.qk_dim)
            .flat_map(|x| norm(x, &qscale))
            .collect();
        let mut k: Vec<f32> = k
            .chunks_exact(g.qk_dim)
            .flat_map(|x| norm(x, &kscale))
            .collect();
        let mut v = v;
        for x in q.chunks_exact_mut(g.qk_dim) {
            rope(x, pos, g.rope_theta);
            quant(x);
        }
        for x in k.chunks_exact_mut(g.qk_dim) {
            rope(x, pos, g.rope_theta);
            quant(x);
        }
        for x in v.chunks_exact_mut(g.v_dim) {
            quant(x);
        }
        Ok((q, k, v))
    }

    fn mix_attention(
        &self,
        layer: usize,
        pos: usize,
        state: &mut LayerState,
        projected: &ProjectedAttention<'_>,
    ) -> Result<Vec<f32>> {
        let g = &self.geometry;
        let (q, k, v) = self.staged_qkv(layer, pos, state, projected)?;
        let window = self.attention_window(layer, pos + 1);
        push_recent(&mut state.keys, k, window);
        push_recent(&mut state.values, v, window);
        let mut out = vec![0.0; g.heads * g.v_dim];
        for head in 0..g.heads {
            let kh = head / (g.heads / g.kv_heads);
            let query = &q[head * g.qk_dim..(head + 1) * g.qk_dim];
            let mut scores: Vec<f32> = (0..state.keys.len())
                .map(|t| {
                    dot(query, &state.keys[t][kh * g.qk_dim..(kh + 1) * g.qk_dim])
                        / (g.qk_dim as f32).sqrt()
                })
                .collect();
            softmax(&mut scores);
            let width = g.v_dim;
            let src = kh * width;
            let dst = head * width;
            for (j, &p) in scores.iter().enumerate() {
                let row = state
                    .values
                    .get(j)
                    .ok_or_else(|| Error::Request("attention value width mismatch".into()))?;
                let values = row
                    .get(src..src + width)
                    .ok_or_else(|| Error::Request("attention value width mismatch".into()))?;
                axpy(&mut out[dst..dst + width], p, values);
            }
        }
        for (value, gate) in out.iter_mut().zip(projected.gates) {
            *value *= sigmoid(*gate);
        }
        quant(&mut out);
        Ok(out)
    }

    /// One product of queries against the keys they can see. `rows` and `keys`
    /// are row-major, and each row has `width` lanes.
    fn dot_matrix(&self, rows: &[f32], keys: &[f32], width: usize) -> Result<Vec<f32>> {
        if width == 0 || !rows.len().is_multiple_of(width) || !keys.len().is_multiple_of(width) {
            return Err(Error::Request("attention width mismatch".into()));
        }
        let queries = rows.len() / width;
        let key_count = keys.len() / width;
        if queries == 0 || key_count == 0 {
            return Ok(Vec::new());
        }
        let device = self.runtime.device().tensor_device();
        let left = Tensor::from_slice(rows, (queries, width), device)?;
        let right = Tensor::from_slice(keys, (key_count, width), device)?;
        let scores = left.matmul(&right.t()?)?.flatten_all()?.to_vec1::<f32>()?;
        if scores.len() != queries * key_count {
            return Err(Error::Request("attention width mismatch".into()));
        }
        Ok(scores)
    }

    /// `weights` is row-major `[rows, keys]` and `values` is `[keys, width]`.
    /// One product replaces a per-key scaled add. Columns outside a query's
    /// window stay zero, so they do not change a finite mix.
    fn mix_values(
        &self,
        weights: &[f32],
        values: &[f32],
        rows: usize,
        keys: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let weight_len = rows
            .checked_mul(keys)
            .ok_or_else(|| Error::Request("attention value width mismatch".into()))?;
        let value_len = keys
            .checked_mul(width)
            .ok_or_else(|| Error::Request("attention value width mismatch".into()))?;
        let out_len = rows
            .checked_mul(width)
            .ok_or_else(|| Error::Request("attention value width mismatch".into()))?;
        if rows == 0
            || keys == 0
            || width == 0
            || weights.len() != weight_len
            || values.len() != value_len
        {
            return Err(Error::Request("attention value width mismatch".into()));
        }
        let device = self.runtime.device().tensor_device();
        let left = Tensor::from_slice(weights, (rows, keys), device)?;
        let right = Tensor::from_slice(values, (keys, width), device)?;
        let mixed = left.matmul(&right)?.flatten_all()?.to_vec1::<f32>()?;
        if mixed.len() != out_len {
            return Err(Error::Request("attention value width mismatch".into()));
        }
        Ok(mixed)
    }

    /// Long-prompt attention. Convolution and the KV cache stay in token order.
    /// Each kv head scores one matrix product per chunk of 256 tokens, then
    /// mixes that chunk's values as one product of the softmax weights.
    fn mix_prefill_attention(
        &self,
        layer: usize,
        base: usize,
        state: &mut LayerState,
        projected: &ProjectedAttention<'_>,
        count: usize,
        cancel: &CancellationToken,
    ) -> Result<Vec<f32>> {
        let g = &self.geometry;
        let group = g.heads.checked_div(g.kv_heads).unwrap_or(0);
        if g.kv_heads == 0
            || group == 0
            || g.qk_dim == 0
            || g.v_dim == 0
            || g.heads != group * g.kv_heads
        {
            return Err(Error::Request("attention width mismatch".into()));
        }
        let nq = g.heads * g.qk_dim;
        let nk = g.kv_heads * g.qk_dim;
        let nv = g.kv_heads * g.v_dim;
        let ng = g.heads * g.v_dim;
        if projected.q.len() != count * nq
            || projected.k.len() != count * nk
            || projected.v.len() != count * nv
            || projected.gates.len() != count * ng
        {
            return Err(Error::Request("projection width mismatch".into()));
        }
        let mut key_log = Vec::new();
        let mut value_log = Vec::new();
        for row in &state.keys {
            if row.len() != nk {
                return Err(Error::Request("attention value width mismatch".into()));
            }
            key_log.extend_from_slice(row);
        }
        for row in &state.values {
            if row.len() != nv {
                return Err(Error::Request("attention value width mismatch".into()));
            }
            value_log.extend_from_slice(row);
        }
        if state.keys.len() != state.values.len() {
            return Err(Error::Request("attention value width mismatch".into()));
        }
        let past = state.keys.len();
        let mut windows = Vec::with_capacity(count);
        for t in 0..count {
            let pos = base
                .checked_add(t)
                .ok_or_else(|| Error::Request("attention index overflow".into()))?;
            let available = pos
                .checked_add(1)
                .ok_or_else(|| Error::Request("attention index overflow".into()))?;
            windows.push(self.attention_window(layer, available));
        }
        let spans = record_key_spans(past, &windows)?;
        let mut queries = Vec::with_capacity(count * nq);
        for (t, &window) in windows.iter().enumerate() {
            check_cancel(cancel)?;
            let (q, k, v) = self.staged_qkv(
                layer,
                base.checked_add(t)
                    .ok_or_else(|| Error::Request("attention index overflow".into()))?,
                state,
                &ProjectedAttention {
                    q: row_at(projected.q, t, nq)?,
                    k: row_at(projected.k, t, nk)?,
                    v: row_at(projected.v, t, nv)?,
                    gates: row_at(projected.gates, t, ng)?,
                },
            )?;
            if q.len() != nq || k.len() != nk || v.len() != nv {
                return Err(Error::Request("attention width mismatch".into()));
            }
            queries.extend_from_slice(&q);
            key_log.extend_from_slice(&k);
            value_log.extend_from_slice(&v);
            push_recent(&mut state.keys, k, window);
            push_recent(&mut state.values, v, window);
        }
        let mut out = vec![0.0; count * ng];
        let mut token = 0;
        while token < count {
            check_cancel(cancel)?;
            let n = 256.min(count - token);
            self.mix_score_chunk(
                &mut out,
                &AttentionLog {
                    queries: &queries,
                    keys: &key_log,
                    values: &value_log,
                    spans: &spans,
                },
                token,
                n,
            )?;
            token += n;
        }
        for (t, row) in out.chunks_exact_mut(ng).enumerate() {
            let gates = row_at(projected.gates, t, ng)?;
            for (value, gate) in row.iter_mut().zip(gates) {
                *value *= sigmoid(*gate);
            }
            quant(row);
        }
        Ok(out)
    }

    fn mix_score_chunk(
        &self,
        out: &mut [f32],
        log: &AttentionLog<'_>,
        token: usize,
        count: usize,
    ) -> Result<()> {
        let g = &self.geometry;
        let group = g.heads.checked_div(g.kv_heads).unwrap_or(0);
        if g.kv_heads == 0 || group == 0 || g.qk_dim == 0 {
            return Err(Error::Request("attention width mismatch".into()));
        }
        let qk = g.qk_dim;
        let nq = g.heads * qk;
        let nk = g.kv_heads * qk;
        let nv = g.kv_heads * g.v_dim;
        let ng = g.heads * g.v_dim;
        let span_end = token
            .checked_add(count)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        let chunk = log
            .spans
            .get(token..span_end)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        let union_start = chunk
            .iter()
            .map(|span| span.0)
            .min()
            .ok_or_else(|| Error::Request("attention width mismatch".into()))?;
        let union_end = chunk
            .iter()
            .map(|span| span.1)
            .max()
            .ok_or_else(|| Error::Request("attention width mismatch".into()))?;
        if union_end < union_start {
            return Err(Error::Request("attention index overflow".into()));
        }
        let union = union_end - union_start;
        if union == 0 {
            return Err(Error::Request("attention width mismatch".into()));
        }
        let scale = (qk as f32).sqrt();
        let rows = count
            .checked_mul(group)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        let weight_len = rows
            .checked_mul(union)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        let value_len = union
            .checked_mul(g.v_dim)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        let mut packed_q = Vec::with_capacity(count * group * qk);
        let mut packed_k = Vec::with_capacity(union * qk);
        let mut weights = vec![0.0f32; weight_len];
        let mut packed_values = Vec::with_capacity(value_len);
        let mut scores_row = Vec::new();
        for kh in 0..g.kv_heads {
            packed_q.clear();
            packed_k.clear();
            for local in 0..count {
                let query = row_at(log.queries, token + local, nq)?;
                for head in 0..group {
                    let src = (kh * group + head) * qk;
                    packed_q.extend_from_slice(
                        query
                            .get(src..src + qk)
                            .ok_or_else(|| Error::Request("attention width mismatch".into()))?,
                    );
                }
            }
            for index in union_start..union_end {
                let row = row_at(log.keys, index, nk)?;
                let src = kh * qk;
                packed_k.extend_from_slice(
                    row.get(src..src + qk)
                        .ok_or_else(|| Error::Request("attention width mismatch".into()))?,
                );
            }
            let scores = self.dot_matrix(&packed_q, &packed_k, qk)?;
            weights.fill(0.0);
            for (local, &(start, end)) in chunk.iter().enumerate() {
                if start < union_start || end > union_end || start > end {
                    return Err(Error::Request("attention index overflow".into()));
                }
                let rel_start = start - union_start;
                let rel_end = end - union_start;
                for head in 0..group {
                    let row_index = local * group + head;
                    let row = row_at(&scores, row_index, union)?;
                    scores_row.clear();
                    scores_row.extend(row[rel_start..rel_end].iter().map(|value| value / scale));
                    softmax(&mut scores_row);
                    let at = row_index
                        .checked_mul(union)
                        .and_then(|base| base.checked_add(rel_start))
                        .ok_or_else(|| Error::Request("attention index overflow".into()))?;
                    let to = at
                        .checked_add(scores_row.len())
                        .ok_or_else(|| Error::Request("attention index overflow".into()))?;
                    let dest = weights
                        .get_mut(at..to)
                        .ok_or_else(|| Error::Request("attention value width mismatch".into()))?;
                    dest.copy_from_slice(&scores_row);
                }
            }
            packed_values.clear();
            for index in union_start..union_end {
                let value = row_at(log.values, index, nv)?;
                let src = kh * g.v_dim;
                packed_values.extend_from_slice(
                    value
                        .get(src..src + g.v_dim)
                        .ok_or_else(|| Error::Request("attention value width mismatch".into()))?,
                );
            }
            let mixed = self.mix_values(&weights, &packed_values, rows, union, g.v_dim)?;
            for local in 0..count {
                for head in 0..group {
                    let row_index = local * group + head;
                    let src = row_at(&mixed, row_index, g.v_dim)?;
                    let dst = (token + local) * ng + (kh * group + head) * g.v_dim;
                    let slot = out
                        .get_mut(dst..dst + g.v_dim)
                        .ok_or_else(|| Error::Request("attention width mismatch".into()))?;
                    slot.copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    fn kron(&self, x: &[f32], a: usize, b: usize) -> Result<Vec<f32>> {
        let aw = self.flat(a)?;
        let bw = self.flat(b)?;
        let na = self.weights[a].dims()[0];
        let nb = self.weights[b].dims()[0];
        Ok(kron_factors(x, &aw, &bw, na, nb))
    }

    fn mlp(&self, x: &[f32], o: usize) -> Result<Vec<f32>> {
        let mut cond = self.affine_input(o + 22, x)?;
        softmax(&mut cond);
        let cond = self.affine_input(o + 23, &cond)?;
        let d1 = self.flat(o + 11)?;
        let d2 = self.flat(o + 12)?;
        let b2 = self.flat(o + 13)?;
        let d3 = self.flat(o + 14)?;
        let d4 = self.flat(o + 15)?;
        let mut z = vec![0.0; self.geometry.hadamard];
        for i in 0..x.len() {
            z[i] = x[i] * d1[i];
        }
        let z = self.kron(&z, o + 16, o + 17)?;
        let z: Vec<f32> = self
            .p1
            .iter()
            .enumerate()
            .map(|(i, &j)| {
                let v = d2[i] * (1.0 + cond[i]) * z[j] + b2[i];
                v * sigmoid(v)
            })
            .collect();
        let z = self.kron(&z, o + 18, o + 19)?;
        let z: Vec<f32> = self
            .p2
            .iter()
            .enumerate()
            .map(|(i, &j)| d3[i] * z[j])
            .collect();
        let z = self.kron(&z, o + 20, o + 21)?;
        Ok(z.iter().zip(d4).take(x.len()).map(|(v, d)| v * d).collect())
    }

    fn prepare_carry(
        &self,
        stream: &[f32],
        layer: usize,
        ap: &[f32],
        ao: &[f32],
        ar: &[f32],
    ) -> Result<TokenCarry> {
        let d = self.geometry.width;
        let n = self.geometry.lanes;
        let mut nx = rms(stream);
        quant(&mut nx);
        let mut pre =
            self.linear_tensor(&self.weights[self.mhc + 6].narrow(0, layer * n, n)?, &nx)?;
        let mut post =
            self.linear_tensor(&self.weights[self.mhc + 7].narrow(0, layer * n, n)?, &nx)?;
        let mut res = self.linear_tensor(
            &self.weights[self.mhc + 8].narrow(0, layer * n * n, n * n)?,
            &nx,
        )?;
        let bp = self.row(self.mhc + 3, layer)?;
        let bo = self.row(self.mhc + 4, layer)?;
        let br = self.row(self.mhc + 5, layer)?;
        for i in 0..n {
            let lane = if i == layer % n { 1.0 } else { 0.0 };
            pre[i] = sigmoid(ap[layer] * pre[i] + bp[i] + 8.0 * lane - 4.0);
            post[i] = 2.0 * sigmoid(ao[layer] * post[i] + bo[i] - 4.0 * (1.0 - lane));
        }
        for i in 0..n * n {
            res[i] = ar[layer] * res[i] + br[i];
        }
        sinkhorn(&mut res, n);
        let mut u = vec![0.0; d];
        for lane in 0..n {
            for i in 0..d {
                u[i] += pre[lane] * stream[lane * d + i];
            }
        }
        Ok(TokenCarry {
            x: u.clone(),
            u,
            post,
            res,
        })
    }

    /// Walk `tokens` from the current state. Long prompts project each layer
    /// as one matrix. The last distribution is the only one returned.
    fn prefill(
        &self,
        tokens: &[u32],
        state: &mut State,
        cancel: &CancellationToken,
        return_last: bool,
    ) -> Result<Vec<f32>> {
        if !self.batched_prefill(tokens.len()) {
            let mut logits = Vec::new();
            let last = tokens.len().saturating_sub(1);
            for (index, &token) in tokens.iter().enumerate() {
                let mode = if return_last && index == last {
                    StepLogits::Return
                } else {
                    StepLogits::Omit
                };
                logits = self.step(token, state, cancel, mode)?;
            }
            return Ok(logits);
        }
        check_cancel(cancel)?;
        let g = &self.geometry;
        let d = g.width;
        let n = g.lanes;
        let count = tokens.len();
        let base = state.tokens.len();
        if state.capture_layers {
            state.layer_means.clear();
        }
        let scale = (d as f32).sqrt();
        let mut streams = Vec::with_capacity(count);
        let mut token_cells = Vec::new();
        for &token in tokens {
            check_cancel(cancel)?;
            let embedding: Vec<f32> = self
                .row(0, token as usize)?
                .iter()
                .map(|value| value * scale)
                .collect();
            streams.push(embedding.repeat(n));
            if state.collect_cells {
                token_cells.push(embedding);
            }
            state.tokens.push(token);
        }
        let ap = self.flat(self.mhc)?;
        let ao = self.flat(self.mhc + 1)?;
        let ar = self.flat(self.mhc + 2)?;
        let nq = g.heads * g.qk_dim;
        let nk = g.kv_heads * g.qk_dim;
        let nv = g.kv_heads * g.v_dim;
        let ng = g.heads * g.v_dim;
        for layer in 0..g.layers {
            check_cancel(cancel)?;
            let mut carries = Vec::with_capacity(count);
            for stream in &streams {
                carries.push(self.prepare_carry(stream, layer, &ap, &ao, &ar)?);
            }
            if let Some(site) = g.engram_sites.iter().position(|&value| value == layer) {
                let mut packed_e = Vec::new();
                for t in 0..count {
                    check_cancel(cancel)?;
                    packed_e.extend(self.engram_embedding(site, &state.tokens, base + t)?);
                }
                let e_len = packed_e.len() / count;
                let weight = self.engrams + 4 * site;
                let keys = self.project_matrix(&self.weights[weight + 1], &packed_e, e_len)?;
                let values = self.project_matrix(&self.weights[weight + 2], &packed_e, e_len)?;
                if !keys.len().is_multiple_of(count) || !values.len().is_multiple_of(count) {
                    return Err(Error::Request("projection width mismatch".into()));
                }
                let k_width = keys.len() / count;
                let v_width = values.len() / count;
                if k_width != d || v_width != d {
                    return Err(Error::Request("projection width mismatch".into()));
                }
                for (t, carry) in carries.iter_mut().enumerate() {
                    let key = row_at(&keys, t, k_width)?.to_vec();
                    let value = row_at(&values, t, v_width)?.to_vec();
                    let (key, value) = self.push_engram(site, state, key, value)?;
                    let mix = sigmoid(dot(&rms(&carry.x), &rms(&key)) / scale);
                    for (slot, term) in carry.x.iter_mut().zip(&value) {
                        *slot += mix * term;
                    }
                }
            }
            let b = self.base(layer);
            let o = b + if g.conv_taps > 0 { 3 } else { 0 };
            let norm_w = self.flat(b)?;
            let mut packed_x = Vec::with_capacity(count * d);
            for carry in &carries {
                let mut input = norm(&carry.x, &norm_w);
                quant(&mut input);
                packed_x.extend(input);
            }
            let q = self.project_matrix(&self.weights[b + 1], &packed_x, d)?;
            let k = self.project_matrix(&self.weights[b + 2], &packed_x, d)?;
            let v = self.project_matrix(&self.weights[b + 3], &packed_x, d)?;
            let gates = self.project_matrix(&self.weights[o + 6], &packed_x, d)?;
            if q.len() != count * nq
                || k.len() != count * nk
                || v.len() != count * nv
                || gates.len() != count * ng
            {
                return Err(Error::Request("projection width mismatch".into()));
            }
            let packed_mix = self.mix_prefill_attention(
                layer,
                base,
                &mut state.layers[layer],
                &ProjectedAttention {
                    q: &q,
                    k: &k,
                    v: &v,
                    gates: &gates,
                },
                count,
                cancel,
            )?;
            if packed_mix.len() != count * ng {
                return Err(Error::Request("projection width mismatch".into()));
            }
            let projected = self.project_matrix(&self.weights[o + 7], &packed_mix, ng)?;
            if projected.len() != count * d {
                return Err(Error::Request("projection width mismatch".into()));
            }
            let att_norm = self.flat(o + 8)?;
            let att_gate = self.flat(o + 9)?;
            let gate = sigmoid(
                *att_gate
                    .first()
                    .ok_or_else(|| Error::Request("missing attention gate".into()))?,
            );
            let mlp_norm = self.flat(o + 10)?;
            for t in 0..count {
                check_cancel(cancel)?;
                let mut att = norm(row_at(&projected, t, d)?, &att_norm);
                for (slot, value) in carries[t].x.iter_mut().zip(&att) {
                    *slot += gate * value;
                }
                att = norm(&carries[t].x, &mlp_norm);
                let mlp = self.mlp(&att, o)?;
                let mut next = vec![0.0; n * d];
                for lane in 0..n {
                    for i in 0..d {
                        next[lane * d + i] =
                            carries[t].post[lane] * (carries[t].x[i] + mlp[i] - carries[t].u[i]);
                        for j in 0..n {
                            next[lane * d + i] +=
                                carries[t].res[lane * n + j] * streams[t][j * d + i];
                        }
                    }
                }
                streams[t] = next;
                if state.collect_cells || (state.capture_layers && t + 1 == count) {
                    let mut mean = vec![0.0; d];
                    for i in 0..d {
                        mean[i] =
                            (0..n).map(|lane| streams[t][lane * d + i]).sum::<f32>() / n as f32;
                    }
                    if state.capture_layers && t + 1 == count {
                        state.layer_means.push(mean.clone());
                    }
                    if state.collect_cells {
                        token_cells[t].extend(mean);
                    }
                }
            }
        }
        if state.collect_cells {
            state.cells.extend(token_cells);
        }
        let final_norm = self.flat(self.final_norm)?;
        let mut last_hidden = Vec::new();
        for (t, stream) in streams.iter().enumerate() {
            check_cancel(cancel)?;
            let mut hidden = vec![0.0; d];
            for lane in 0..n {
                for i in 0..d {
                    hidden[i] += stream[lane * d + i] / n as f32;
                }
            }
            let mut hidden = norm(&hidden, &final_norm);
            quant(&mut hidden);
            if hidden.iter().any(|value| !value.is_finite()) {
                return Err(Error::Request("non-finite model output".into()));
            }
            if state.capture_layers && t + 1 == count {
                state.quantized_hidden.clone_from(&hidden);
            }
            if t + 1 == count {
                last_hidden = hidden;
            }
        }
        if !return_last {
            return Ok(Vec::new());
        }
        let logits = if g.out_vocab == g.vocab {
            self.gemv(&self.weights[0], &last_hidden)?
        } else {
            let head = self.weights[0].narrow(1, 0, g.out_vocab)?.contiguous()?;
            self.gemv(&head, &last_hidden)?
        };
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(Error::Request("non-finite model output".into()));
        }
        check_cancel(cancel)?;
        Ok(logits)
    }

    fn step(
        &self,
        token: u32,
        state: &mut State,
        cancel: &CancellationToken,
        logits_mode: StepLogits,
    ) -> Result<Vec<f32>> {
        check_cancel(cancel)?;
        if state.capture_layers {
            state.layer_means.clear();
        }
        let g = &self.geometry;
        let d = g.width;
        let n = g.lanes;
        let pos = state.tokens.len();
        state.tokens.push(token);
        let embedding: Vec<f32> = self
            .row(0, token as usize)?
            .iter()
            .map(|v| v * (d as f32).sqrt())
            .collect();
        let mut stream = embedding.repeat(n);
        let mut cells = if state.collect_cells {
            embedding
        } else {
            Vec::new()
        };
        let ap = self.flat(self.mhc)?;
        let ao = self.flat(self.mhc + 1)?;
        let ar = self.flat(self.mhc + 2)?;
        for layer in 0..g.layers {
            check_cancel(cancel)?;
            let mut nx = rms(&stream);
            quant(&mut nx);
            let mut pre =
                self.linear_tensor(&self.weights[self.mhc + 6].narrow(0, layer * n, n)?, &nx)?;
            let mut post =
                self.linear_tensor(&self.weights[self.mhc + 7].narrow(0, layer * n, n)?, &nx)?;
            let mut res = self.linear_tensor(
                &self.weights[self.mhc + 8].narrow(0, layer * n * n, n * n)?,
                &nx,
            )?;
            let bp = self.row(self.mhc + 3, layer)?;
            let bo = self.row(self.mhc + 4, layer)?;
            let br = self.row(self.mhc + 5, layer)?;
            for i in 0..n {
                let lane = if i == layer % n { 1.0 } else { 0.0 };
                pre[i] = sigmoid(ap[layer] * pre[i] + bp[i] + 8.0 * lane - 4.0);
                post[i] = 2.0 * sigmoid(ao[layer] * post[i] + bo[i] - 4.0 * (1.0 - lane));
            }
            for i in 0..n * n {
                res[i] = ar[layer] * res[i] + br[i];
            }
            sinkhorn(&mut res, n);
            let mut u = vec![0.0; d];
            for lane in 0..n {
                for i in 0..d {
                    u[i] += pre[lane] * stream[lane * d + i];
                }
            }
            let mut x = u.clone();
            if let Some(site) = g.engram_sites.iter().position(|&v| v == layer) {
                let (k, v) = self.engram(site, state, pos)?;
                let a = sigmoid(dot(&rms(&x), &rms(&k)) / (d as f32).sqrt());
                for i in 0..d {
                    x[i] += a * v[i];
                }
            }
            let b = self.base(layer);
            let o = b + if g.conv_taps > 0 { 3 } else { 0 };
            let input = norm(&x, &self.flat(b)?);
            let att = self.attention(&input, layer, pos, &mut state.layers[layer])?;
            let att = norm(&att, &self.flat(o + 8)?);
            let gate = sigmoid(self.flat(o + 9)?[0]);
            for i in 0..d {
                x[i] += gate * att[i];
            }
            let mlp = self.mlp(&norm(&x, &self.flat(o + 10)?), o)?;
            let mut next = vec![0.0; n * d];
            for lane in 0..n {
                for i in 0..d {
                    next[lane * d + i] = post[lane] * (x[i] + mlp[i] - u[i]);
                    for j in 0..n {
                        next[lane * d + i] += res[lane * n + j] * stream[j * d + i];
                    }
                }
            }
            stream = next;
            if state.capture_layers {
                let mut mean = vec![0.0; d];
                for i in 0..d {
                    mean[i] = (0..n).map(|lane| stream[lane * d + i]).sum::<f32>() / n as f32;
                }
                state.layer_means.push(mean);
            }
            if state.collect_cells {
                for i in 0..d {
                    cells.push((0..n).map(|lane| stream[lane * d + i]).sum::<f32>() / n as f32);
                }
            }
        }
        let mut hidden = vec![0.0; d];
        for lane in 0..n {
            for i in 0..d {
                hidden[i] += stream[lane * d + i] / n as f32;
            }
        }
        let mut hidden = norm(&hidden, &self.flat(self.final_norm)?);
        quant(&mut hidden);
        if state.capture_layers {
            state.quantized_hidden.clone_from(&hidden);
        }
        if logits_mode == StepLogits::Omit {
            if hidden.iter().any(|value| !value.is_finite()) {
                return Err(Error::Request("non-finite model output".into()));
            }
            check_cancel(cancel)?;
            if state.collect_cells {
                state.cells.push(cells);
            }
            return Ok(Vec::new());
        }
        let logits = if g.out_vocab == g.vocab {
            self.gemv(&self.weights[0], &hidden)?
        } else {
            let head = self.weights[0].narrow(1, 0, g.out_vocab)?.contiguous()?;
            self.gemv(&head, &hidden)?
        };
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(Error::Request("non-finite model output".into()));
        }
        check_cancel(cancel)?;
        if state.collect_cells {
            state.cells.push(cells);
        }
        Ok(logits)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepLogits {
    /// Last-token distribution. Prefill and readouts discard every earlier one.
    Return,
    Omit,
}

fn full_projection_indices(g: &Geometry) -> Vec<usize> {
    let stride = if g.conv_taps > 0 { 27 } else { 24 };
    let mut ids = Vec::with_capacity(1 + g.layers * 5 + g.engram_sites.len() * 2);
    ids.push(0);
    for layer in 0..g.layers {
        let b = 1 + layer * stride;
        let o = b + if g.conv_taps > 0 { 3 } else { 0 };
        ids.extend([b + 1, b + 2, b + 3, o + 6, o + 7]);
    }
    let engrams = 1 + stride * g.layers + 11;
    for site in 0..g.engram_sites.len() {
        let b = engrams + 4 * site;
        ids.extend([b + 1, b + 2]);
    }
    ids
}

fn transpose_out_in_inplace(values: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    if rows.checked_mul(cols) != Some(values.len()) {
        return Err(Error::Archive("projection shape mismatch".into()));
    }
    if rows == cols {
        for row in 0..rows {
            for col in row + 1..cols {
                values.swap(row * cols + col, col * rows + row);
            }
        }
        return Ok(());
    }
    if rows <= 1 || cols <= 1 {
        return Ok(());
    }
    // Follow the permutation in place. A bit set marks visited indices so a
    // non-square matrix does not need a second copy of the weights.
    let n = values.len();
    let mut seen = vec![0u64; n.div_ceil(64)];
    for start in 0..n {
        if seen[start / 64] & (1u64 << (start % 64)) != 0 {
            continue;
        }
        let mut src = start;
        let mut value = values[start];
        loop {
            let dst = (src % cols) * rows + src / cols;
            seen[src / 64] |= 1u64 << (src % 64);
            if dst == start {
                values[dst] = value;
                break;
            }
            value = std::mem::replace(&mut values[dst], value);
            src = dst;
        }
    }
    Ok(())
}

fn orient_tensor(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(Error::Archive("projection rank mismatch".into()));
    }
    let mut values = tensor.flatten_all()?.to_vec1::<f32>()?;
    transpose_out_in_inplace(&mut values, dims[0], dims[1])?;
    Ok(Tensor::from_vec(
        values,
        &[dims[1], dims[0]],
        tensor.device(),
    )?)
}

fn row_at(values: &[f32], index: usize, width: usize) -> Result<&[f32]> {
    let start = index
        .checked_mul(width)
        .ok_or_else(|| Error::Request("projection width mismatch".into()))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| Error::Request("projection width mismatch".into()))?;
    values
        .get(start..end)
        .ok_or_else(|| Error::Request("projection width mismatch".into()))
}

fn push_recent(history: &mut VecDeque<Vec<f32>>, value: Vec<f32>, capacity: usize) {
    if history.len() == capacity {
        history.pop_front();
    }
    history.push_back(value);
}

/// Key-log spans produced by the same pop-front rule as `push_recent`.
/// `past` is the number of rows already cached. Each window is the capacity
/// used when that token is appended.
fn record_key_spans(past: usize, windows: &[usize]) -> Result<Vec<(usize, usize)>> {
    let mut start = 0usize;
    let mut len = past;
    let mut spans = Vec::with_capacity(windows.len());
    for &window in windows {
        if len == window && len > 0 {
            start = start
                .checked_add(1)
                .ok_or_else(|| Error::Request("attention index overflow".into()))?;
            len -= 1;
        }
        len = len
            .checked_add(1)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::Request("attention index overflow".into()))?;
        spans.push((start, end));
    }
    Ok(spans)
}

fn add_bytes(total: &mut u64, factors: &[usize]) -> Result<()> {
    let bytes = factors
        .iter()
        .try_fold(1u64, |n, &v| n.checked_mul(v as u64));
    *total = bytes
        .and_then(|n| total.checked_add(n))
        .ok_or_else(|| Error::Request("state size overflow".into()))?;
    Ok(())
}

/// Kronecker product of two square factors. Each lane is one multiply and one
/// add, in the same loop order as the scalar triple loop.
fn kron_factors(x: &[f32], aw: &[f32], bw: &[f32], na: usize, nb: usize) -> Vec<f32> {
    let mut mid = vec![0.0; x.len()];
    let mut out = vec![0.0; x.len()];
    for i in 0..na {
        let dst = i * nb;
        for j in 0..nb {
            let scale = x[dst + j];
            let row = j * nb;
            axpy(&mut mid[dst..dst + nb], scale, &bw[row..row + nb]);
        }
    }
    for i in 0..na {
        let src = i * nb;
        let col = i * na;
        for k in 0..na {
            axpy(
                &mut out[k * nb..k * nb + nb],
                aw[col + k],
                &mid[src..src + nb],
            );
        }
    }
    out
}

pub(crate) fn check_cancel(cancel: &CancellationToken) -> Result<()> {
    if process_stop_requested() {
        cancel.cancel();
    }
    if cancel.is_cancelled() {
        Err(a3s_power::error::PowerError::InferenceCancelled.into())
    } else {
        Ok(())
    }
}

pub(crate) fn checkpoint(cancel: &CancellationToken, stage: &'static str) -> Result<()> {
    #[cfg(test)]
    crate::test_support::hit(stage, cancel);
    #[cfg(not(test))]
    let _ = stage;
    check_cancel(cancel)
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
