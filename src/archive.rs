use std::{fs::File, io::Read, path::Path};

use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{Error, InferenceLimits, Result, Tokenizer};

const TAG: u32 = 0x05e12a84;
const HEADER: usize = 196;
const RECORD: usize = 44;

#[derive(Clone, Debug, Serialize)]
pub struct Geometry {
    pub vocab: usize,
    pub out_vocab: usize,
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub layers: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub context: usize,
    pub hadamard: usize,
    pub lanes: usize,
    pub sliding_window: usize,
    pub global_mask: u64,
    pub conv_taps: usize,
    pub engram_slots: usize,
    pub engram_dim: usize,
    pub engram_tables: usize,
    pub engram_taps: usize,
    pub engram_dilation: usize,
    pub engram_seed_heads: usize,
    pub engram_orders: Vec<usize>,
    pub engram_sites: Vec<usize>,
    pub rope_theta: f32,
    pub kv_window: usize,
    pub kv_bits: usize,
}

#[derive(Debug)]
pub(crate) struct Record {
    pub dtype: u8,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
    pub group: usize,
    pub bits: usize,
}

pub struct Archive {
    pub(crate) geometry: Geometry,
    pub(crate) sha256: String,
    pub(crate) records: Vec<Record>,
    data: Vec<u8>,
    codebook: Vec<f32>,
}

fn bad(message: impl Into<String>) -> Error {
    Error::Archive(message.into())
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    let b = bytes
        .get(at..at + 4)
        .ok_or_else(|| bad("truncated integer"))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn size_at(bytes: &[u8], at: usize) -> Result<usize> {
    let lo = u32_at(bytes, at)? as u64;
    let hi = u32_at(bytes, at + 4)? as u64;
    usize::try_from(lo | hi << 32).map_err(|_| bad("offset exceeds address space"))
}

impl Archive {
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn open(path: impl AsRef<Path>, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        let file = File::open(path)?;
        if file.metadata()?.len() > limits.max_model_bytes {
            return Err(bad("model byte limit exceeded"));
        }
        let mut bytes = Vec::new();
        file.take(limits.max_model_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        Self::from_bytes(bytes, limits)
    }

    pub fn from_bytes(data: Vec<u8>, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        if data.len() as u64 > limits.max_model_bytes || data.len() < HEADER {
            return Err(bad("invalid model length"));
        }
        let h: Vec<usize> = (0..48)
            .map(|i| u32_at(&data, i * 4).map(|x| x as usize))
            .collect::<Result<_>>()?;
        if h[0] != TAG as usize {
            return Err(bad("unsupported format tag; expected Needle 3"));
        }
        if h[2] != 28 || h[1] == 0 || h[1] > limits.max_graph_initializers {
            return Err(bad("invalid tensor or codebook count"));
        }
        if h[26] > 4 || h[31] > 16 {
            return Err(bad("invalid engram counts"));
        }
        let g = Geometry {
            vocab: h[5],
            out_vocab: if h[6] == 0 { h[5] } else { h[6] },
            width: h[7],
            heads: h[8],
            kv_heads: h[9],
            layers: h[10],
            qk_dim: h[11],
            v_dim: h[12],
            context: h[13],
            hadamard: h[14],
            lanes: h[15],
            sliding_window: h[16],
            global_mask: h[17] as u64 | (h[18] as u64) << 32,
            conv_taps: h[19],
            engram_slots: h[20],
            engram_dim: h[21],
            engram_tables: h[22],
            engram_taps: h[23],
            engram_dilation: h[24],
            engram_seed_heads: h[25],
            engram_orders: h[27..27 + h[26]].to_vec(),
            engram_sites: h[32..32 + h[31]].to_vec(),
            rope_theta: f32::from_bits(u32_at(&data, 192)?),
            kv_window: h[3],
            kv_bits: h[4],
        };
        if [
            g.vocab, g.width, g.heads, g.kv_heads, g.layers, g.qk_dim, g.v_dim, g.context, g.lanes,
        ]
        .contains(&0)
            || g.layers > 64
            || !g.heads.is_multiple_of(g.kv_heads)
            || !g.qk_dim.is_multiple_of(2)
            || g.out_vocab > g.vocab
            || !g.hadamard.is_power_of_two()
            || g.hadamard < g.width
            || !g.rope_theta.is_finite()
            || g.rope_theta <= 0.0
            || g.engram_sites.iter().any(|&i| i >= g.layers)
            || g.engram_sites.windows(2).any(|w| w[0] >= w[1])
            || [
                g.vocab,
                g.width,
                g.qk_dim,
                g.v_dim,
                g.context,
                g.hadamard,
                g.engram_slots,
                g.engram_dim,
            ]
            .iter()
            .any(|&n| n > 1_048_576)
            || [
                g.heads,
                g.kv_heads,
                g.lanes,
                g.conv_taps,
                g.engram_tables,
                g.engram_taps,
                g.engram_dilation,
                g.engram_seed_heads,
            ]
            .iter()
            .any(|&n| n > 256)
            || g.engram_orders.iter().any(|&n| n > 256)
        {
            return Err(bad("invalid model geometry"));
        }
        if !g.engram_sites.is_empty()
            && (g.engram_slots == 0
                || g.engram_orders.is_empty()
                || g.engram_orders.contains(&0)
                || g.engram_tables == 0
                || g.engram_dim == 0
                || !g.engram_tables.is_multiple_of(g.engram_orders.len())
                || g.engram_taps == 0
                || g.engram_dilation == 0)
        {
            return Err(bad("invalid engram geometry"));
        }
        let directory = HEADER + h[2] * 4;
        let end = directory
            .checked_add(
                h[1].checked_mul(RECORD)
                    .ok_or_else(|| bad("directory overflow"))?,
            )
            .ok_or_else(|| bad("directory overflow"))?;
        if end > data.len() {
            return Err(bad("truncated directory"));
        }
        let codebook: Vec<f32> = (0..h[2])
            .map(|i| u32_at(&data, HEADER + i * 4).map(f32::from_bits))
            .collect::<Result<_>>()?;
        if codebook.iter().any(|v| !v.is_finite()) {
            return Err(bad("non-finite codebook"));
        }
        let mut records = Vec::with_capacity(h[1]);
        let mut previous_end = end;
        for i in 0..h[1] {
            let at = directory + i * RECORD;
            let dtype = data[at];
            let ndim = data[at + 1] as usize;
            if ndim > 4 || ![1, 2, 3, 4].contains(&dtype) || data[at + 2..at + 4] != [0, 0] {
                return Err(bad("invalid tensor record"));
            }
            let shape = (0..ndim)
                .map(|j| u32_at(&data, at + 4 + j * 4).map(|v| v as usize))
                .collect::<Result<Vec<_>>>()?;
            let offset = size_at(&data, at + 20)?;
            let bytes = size_at(&data, at + 28)?;
            let group = u32_at(&data, at + 36)? as usize;
            let bits = u32_at(&data, at + 40)? as usize;
            let tensor_end = offset
                .checked_add(bytes)
                .ok_or_else(|| bad("tensor range overflow"))?;
            if offset % 64 != 0 || offset < previous_end || tensor_end > data.len() || bytes == 0 {
                return Err(bad("invalid or overlapping tensor range"));
            }
            let elements = if dtype == 4 {
                0
            } else {
                limits.checked_elements(&shape, "archive tensor")?
            };
            let expected = match dtype {
                1 => elements.checked_mul(2),
                2 => elements.checked_mul(4),
                3 => {
                    if ndim != 2 || group != 128 || ![1, 2, 3, 4, 5].contains(&bits) {
                        return Err(bad("unsupported CQ geometry"));
                    }
                    let padded = shape[1]
                        .checked_add(group - 1)
                        .ok_or_else(|| bad("CQ overflow"))?
                        / group
                        * group;
                    let width = if bits == 5 { 2 } else { bits };
                    padded
                        .checked_mul(width)
                        .and_then(|n| (n / 8).checked_add(padded / group * 2))
                        .and_then(|n| n.checked_mul(shape[0]))
                }
                _ => {
                    if ndim != 0 {
                        return Err(bad("RAW attachment must be scalar"));
                    }
                    Some(bytes)
                }
            };
            if expected != Some(bytes) {
                return Err(bad(format!("tensor {i} byte length mismatch")));
            }
            previous_end = tensor_end;
            records.push(Record {
                dtype,
                shape,
                offset,
                bytes,
                group,
                bits,
            });
        }
        if records.last().is_none_or(|r| r.dtype != 4)
            || records[..records.len() - 1].iter().any(|r| r.dtype == 4)
        {
            return Err(bad("expected one trailing tokenizer"));
        }
        let sha256 = format!("{:x}", Sha256::digest(&data));
        Ok(Self {
            geometry: g,
            sha256,
            records,
            data,
            codebook,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.records.len()
    }

    pub fn tokenizer(&self) -> Result<Tokenizer> {
        let r = self
            .records
            .last()
            .ok_or_else(|| bad("missing tokenizer"))?;
        Tokenizer::from_bytes(
            &self.data[r.offset..r.offset + r.bytes],
            self.geometry.vocab,
        )
    }

    pub(crate) fn values(&self, index: usize) -> Result<Vec<f32>> {
        let r = self
            .records
            .get(index)
            .ok_or_else(|| bad("missing tensor"))?;
        let data = &self.data[r.offset..r.offset + r.bytes];
        let values = match r.dtype {
            1 => data
                .chunks_exact(2)
                .map(|x| f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
                .collect(),
            2 => data
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect(),
            3 => {
                let rows = r.shape[0];
                let cols = r.shape[1];
                let padded = cols.div_ceil(r.group) * r.group;
                let bits = if r.bits == 5 { 2 } else { r.bits };
                let row_bytes = padded * bits / 8;
                let mut out = vec![0.0; rows * cols];
                for row in 0..rows {
                    for group in 0..padded / r.group {
                        let ni = rows * row_bytes + (row * padded / r.group + group) * 2;
                        let norm =
                            f16::from_bits(u16::from_le_bytes([data[ni], data[ni + 1]])).to_f32();
                        let mut values = vec![0.0; r.group];
                        for (j, value) in values.iter_mut().enumerate() {
                            let bit = (group * r.group + j) * bits;
                            let byte = row * row_bytes + bit / 8;
                            let mut word = data[byte] as usize;
                            if bit % 8 + bits > 8 {
                                word |= (data[byte + 1] as usize) << 8;
                            }
                            let idx = (word >> (bit % 8)) & ((1 << bits) - 1);
                            let centroid = match r.bits {
                                1 => {
                                    (if idx == 0 { -1.0 } else { 1.0 })
                                        * (2.0 / std::f32::consts::PI).sqrt()
                                        / (r.group as f32).sqrt()
                                }
                                5 => {
                                    (match idx {
                                        3 => -1.2240064,
                                        0 => 0.0,
                                        1 => 1.2240064,
                                        _ => return Err(bad("invalid ternary crumb")),
                                    }) / (r.group as f32).sqrt()
                                }
                                b => self.codebook[(1 << b) - 4 + idx],
                            };
                            *value = centroid * norm;
                        }
                        hadamard(&mut values);
                        for (j, v) in values.into_iter().enumerate() {
                            let col = group * r.group + j;
                            if col < cols {
                                out[row * cols + col] = v;
                            }
                        }
                    }
                }
                out
            }
            _ => return Err(bad("RAW is not a numeric tensor")),
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(bad("non-finite tensor"));
        }
        Ok(values)
    }
}

pub(crate) fn hadamard(x: &mut [f32]) {
    let mut step = 1;
    while step < x.len() {
        for base in (0..x.len()).step_by(step * 2) {
            for j in 0..step {
                let a = x[base + j];
                let b = x[base + j + step];
                x[base + j] = a + b;
                x[base + j + step] = a - b;
            }
        }
        step *= 2;
    }
    let scale = (x.len() as f32).sqrt();
    for v in x {
        *v /= scale;
    }
}

#[cfg(test)]
#[path = "archive_tests.rs"]
mod tests;
