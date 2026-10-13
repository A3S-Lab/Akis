use crate::math::{dot, norm, quant, rms, rope, sigmoid, sinkhorn, softmax};
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
        let mut bytes = 0u64;
        for rec in &archive.records[..archive.records.len() - 1] {
            let n = runtime
                .limits()
                .checked_elements(&rec.shape, "model tensor")?;
            bytes = bytes
                .checked_add((n as u64) * 4)
                .ok_or_else(|| Error::Archive("resident size overflow".into()))?;
        }
        if bytes > runtime.limits().max_resident_weight_bytes {
            return Err(Error::Archive(
                "dequantized weights exceed Power residency limit".into(),
            ));
        }
        let tokenizer = archive.tokenizer()?;
        // shortcut: CQ matrices expand to F32; add packed kernels before claiming tiny-device memory use.
        let mut weights = Vec::new();
        for i in 0..archive.records.len() - 1 {
            weights.push(Tensor::from_vec(
                archive.values(i)?,
                archive.records[i].shape.as_slice(),
                runtime.device().tensor_device(),
            )?);
        }
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
        let model = Self {
            tokenizer,
            geometry: archive.geometry,
            sha256: archive.sha256,
            runtime,
            weights,
            mhc,
            engrams,
            final_norm,
            p1,
            p2,
            heads,
        };
        model.validate_shapes()?;
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
        if self.weights.get(i).is_none_or(|t| t.dims() != expected) {
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
        Ok(self.weights[i].get(row)?.flatten_all()?.to_vec1::<f32>()?)
    }
    fn linear_tensor(&self, w: &Tensor, x: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_slice(x, (1, x.len()), self.runtime.device().tensor_device())?;
        Ok(x.matmul(&w.t()?)?.flatten_all()?.to_vec1::<f32>()?)
    }
    fn linear(&self, i: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.linear_tensor(&self.weights[i], x)
    }
    fn affine_input(&self, i: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.linear_tensor(&self.weights[i].t()?, x)
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
        }
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
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self.step(token, &mut state, cancel)?;
        }
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
        let mut logits = Vec::new();
        for &token in &input {
            logits = self.step(token, &mut state, cancel)?;
        }
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
            logits = self.step(id as u32, &mut state, cancel)?;
        }
        // Include the final selected token in the post-hoc head's context.
        if finished {
            if let Some(&last) = output.last() {
                self.step(last, &mut state, cancel)?;
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
        for id in ids {
            self.step(id, &mut state, cancel)?;
        }
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

    fn engram(&self, site: usize, state: &mut State) -> Result<(Vec<f32>, Vec<f32>)> {
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
                if state.tokens.len() < order {
                    e.extend(vec![0.0; g.engram_dim]);
                    continue;
                }
                let mut acc = 0x9e3779b9u32.wrapping_mul((oi * stride + head + 1) as u32);
                for j in 0..order {
                    acc = (acc ^ state.tokens[state.tokens.len() - 1 - j]).wrapping_mul(0x01000193);
                }
                acc ^= acc >> 15;
                e.extend(self.row(
                    base,
                    (oi * heads + head) * g.engram_slots + acc as usize % g.engram_slots,
                )?);
            }
        }
        quant(&mut e);
        let k = self.linear(base + 1, &e)?;
        let v = self.linear(base + 2, &e)?;
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
        let history = g.conv_taps.max(1);
        push_recent(&mut state.q, self.linear(b + 1, &x)?, history);
        push_recent(&mut state.k, self.linear(b + 2, &x)?, history);
        push_recent(&mut state.v, self.linear(b + 3, &x)?, history);
        let (q, k, v) = if g.conv_taps > 0 {
            (
                self.tapped(&state.q, b + 4, g.conv_taps, 1)?,
                self.tapped(&state.k, b + 5, g.conv_taps, 1)?,
                self.tapped(&state.v, b + 6, g.conv_taps, 1)?,
            )
        } else {
            (
                state.q[state.q.len() - 1].clone(),
                state.k[state.k.len() - 1].clone(),
                state.v[state.v.len() - 1].clone(),
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
            for (j, p) in scores.iter().enumerate() {
                for i in 0..g.v_dim {
                    out[head * g.v_dim + i] += p * state.values[j][kh * g.v_dim + i];
                }
            }
        }
        let gates = self.linear(o + 6, &x)?;
        for (v, g) in out.iter_mut().zip(gates) {
            *v *= sigmoid(g);
        }
        quant(&mut out);
        self.linear(o + 7, &out)
    }

    fn kron(&self, x: &[f32], a: usize, b: usize) -> Result<Vec<f32>> {
        let aw = self.flat(a)?;
        let bw = self.flat(b)?;
        let na = self.weights[a].dims()[0];
        let nb = self.weights[b].dims()[0];
        let mut mid = vec![0.0; x.len()];
        let mut out = vec![0.0; x.len()];
        for i in 0..na {
            for j in 0..nb {
                for l in 0..nb {
                    mid[i * nb + l] += x[i * nb + j] * bw[j * nb + l];
                }
            }
        }
        for i in 0..na {
            for k in 0..na {
                for l in 0..nb {
                    out[k * nb + l] += aw[i * na + k] * mid[i * nb + l];
                }
            }
        }
        Ok(out)
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

    fn step(&self, token: u32, state: &mut State, cancel: &CancellationToken) -> Result<Vec<f32>> {
        check_cancel(cancel)?;
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
                let (k, v) = self.engram(site, state)?;
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
        let logits = self.linear_tensor(&self.weights[0].narrow(0, 0, g.out_vocab)?, &hidden)?;
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

fn push_recent(history: &mut VecDeque<Vec<f32>>, value: Vec<f32>, capacity: usize) {
    if history.len() == capacity {
        history.pop_front();
    }
    history.push_back(value);
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
