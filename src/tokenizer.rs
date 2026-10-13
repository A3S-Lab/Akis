use crate::{CancellationToken, Error, InferenceLimits, Result};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::convert::Infallible;

struct Symbol {
    start: usize,
    end: usize,
    previous: Option<usize>,
    next: Option<usize>,
    generation: usize,
}

#[derive(PartialEq)]
struct Merge {
    score: f32,
    left: usize,
    right: usize,
    generations: [usize; 2],
}

impl Eq for Merge {}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Equal scores use the leftmost pair, including the +0/-0 tie.
        let score = if self.score == other.score {
            Ordering::Equal
        } else {
            self.score.total_cmp(&other.score)
        };
        score
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| self.generations.cmp(&other.generations))
    }
}

pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    kinds: Vec<u8>,
    lookup: HashMap<String, u32>,
    bytes: HashMap<u8, u32>,
    markers: Vec<String>,
    dummy: bool,
    fallback: bool,
    pub pad: u32,
    pub eos: u32,
    pub bos: u32,
    pub unk: u32,
}

impl Tokenizer {
    pub(crate) fn from_bytes(data: &[u8], vocab: usize) -> Result<Self> {
        let bad = || Error::Tokenizer("malformed SentencePiece attachment".into());
        if data.len() < 24 {
            return Err(bad());
        }
        let read = |at| u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        let count = read(0) as usize;
        if count == 0
            || count > vocab
            || count > (data.len() - 24) / 7
            || data[20] > 1
            || data[21] > 1
        {
            return Err(bad());
        }
        let (pad, eos, bos, unk) = (read(4), read(8), read(12), read(16));
        if [pad, eos, bos, unk].iter().any(|&i| i as usize >= count) {
            return Err(bad());
        }
        let mut pieces = Vec::new();
        let mut scores = Vec::new();
        let mut kinds = Vec::new();
        let mut lookup = HashMap::new();
        let mut bytes = HashMap::new();
        let mut markers = Vec::new();
        let mut at = 24;
        for i in 0..count {
            let rec = data.get(at..at + 7).ok_or_else(bad)?;
            let score = f32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
            let kind = rec[4];
            let len = u16::from_le_bytes([rec[5], rec[6]]) as usize;
            at += 7;
            let piece = std::str::from_utf8(data.get(at..at + len).ok_or_else(bad)?)
                .map_err(|_| bad())?
                .to_string();
            if !score.is_finite()
                || kind > 4
                || piece.is_empty()
                || lookup.insert(piece.clone(), i as u32).is_some()
            {
                return Err(bad());
            }
            if kind == 4 {
                if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
                    return Err(bad());
                }
                let byte = u8::from_str_radix(&piece[3..5], 16).map_err(|_| bad())?;
                if bytes.insert(byte, i as u32).is_some() {
                    return Err(bad());
                }
            }
            if kind == 3 {
                markers.push(piece.clone());
            }
            pieces.push(piece);
            scores.push(score);
            kinds.push(kind);
            at += len;
        }
        if at != data.len() || (data[21] == 1 && bytes.len() != 256) {
            return Err(bad());
        }
        markers.sort_by_key(|m| std::cmp::Reverse(m.chars().count()));
        Ok(Self {
            pieces,
            scores,
            kinds,
            lookup,
            bytes,
            markers,
            dummy: data[20] == 1,
            fallback: data[21] == 1,
            pad,
            eos,
            bos,
            unk,
        })
    }

    fn bpe_controlled<E>(
        &self,
        text: &str,
        out: &mut Vec<u32>,
        poll: &mut impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let mut symbols: Vec<Symbol> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, c))| {
                if i.is_multiple_of(256) {
                    poll()?;
                }
                Ok(Symbol {
                    start,
                    end: start + c.len_utf8(),
                    previous: i.checked_sub(1),
                    next: Some(i + 1),
                    generation: 0,
                })
            })
            .collect::<std::result::Result<_, E>>()?;
        let Some(last) = symbols.last_mut() else {
            return Ok(());
        };
        last.next = None;
        let mut heap = BinaryHeap::new();
        for left in 0..symbols.len() - 1 {
            if left.is_multiple_of(256) {
                poll()?;
            }
            self.push_merge(text, &symbols, left, &mut heap);
        }
        let mut merges = 0usize;
        while let Some(candidate) = heap.pop() {
            if merges.is_multiple_of(256) {
                poll()?;
            }
            merges += 1;
            let (left, right) = (candidate.left, candidate.right);
            if symbols[left].next != Some(right)
                || candidate.generations != [symbols[left].generation, symbols[right].generation]
            {
                continue;
            }
            symbols[left].end = symbols[right].end;
            symbols[left].next = symbols[right].next;
            symbols[left].generation += 1;
            symbols[right].generation += 1;
            if let Some(next) = symbols[left].next {
                symbols[next].previous = Some(left);
            }
            if let Some(previous) = symbols[left].previous {
                self.push_merge(text, &symbols, previous, &mut heap);
            }
            self.push_merge(text, &symbols, left, &mut heap);
        }
        let mut current = Some(0usize);
        let mut emitted = 0usize;
        while let Some(i) = current {
            if emitted.is_multiple_of(256) {
                poll()?;
            }
            emitted += 1;
            let s = &text[symbols[i].start..symbols[i].end];
            if let Some(&id) = self.lookup.get(s) {
                out.push(id);
            } else if self.fallback {
                out.extend(s.bytes().map(|b| self.bytes[&b]));
            } else {
                out.push(self.unk);
            }
            current = symbols[i].next;
        }
        Ok(())
    }

    #[cfg(test)]
    fn bpe(&self, text: &str, out: &mut Vec<u32>) {
        infallible(self.bpe_controlled(text, out, &mut || Ok::<_, Infallible>(())))
    }

    fn push_merge(
        &self,
        text: &str,
        symbols: &[Symbol],
        left: usize,
        heap: &mut BinaryHeap<Merge>,
    ) {
        let Some(right) = symbols[left].next else {
            return;
        };
        if let Some(&id) = self
            .lookup
            .get(&text[symbols[left].start..symbols[right].end])
        {
            heap.push(Merge {
                score: self.scores[id as usize],
                left,
                right,
                generations: [symbols[left].generation, symbols[right].generation],
            });
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        infallible(self.encode_controlled(text, &mut || Ok::<_, Infallible>(())))
    }

    /// Enforces Power input/state limits and polls cancellation throughout BPE.
    /// `encode` remains an unbounded pure utility for trusted caller-owned data.
    pub fn encode_bounded(
        &self,
        text: &str,
        limits: &InferenceLimits,
        cancel: &CancellationToken,
    ) -> Result<Vec<u32>> {
        crate::model::check_cancel(cancel)?;
        limits.validate()?;
        if text.len() > limits.max_input_bytes {
            return Err(Error::Request("input byte limit exceeded".into()));
        }
        let n = text.len() as u64 + 8;
        // At most 3N queued pairs, Vec growth <2x, plus symbols, bytes and IDs.
        let per_byte = 2 * std::mem::size_of::<Symbol>() + 6 * std::mem::size_of::<Merge>() + 40;
        let bytes = n
            .checked_mul(per_byte as u64)
            .ok_or_else(|| Error::Request("tokenizer state size overflow".into()))?;
        limits.checked_state_bytes(bytes, "tokenizer state")?;
        self.encode_controlled(text, &mut || crate::model::check_cancel(cancel))
    }

    fn encode_controlled<E>(
        &self,
        text: &str,
        poll: &mut impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Vec<u32>, E> {
        poll()?;
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let escaped = text.replace(' ', "▁");
        let escaped = if self.dummy {
            format!("▁{escaped}")
        } else {
            escaped
        };
        let mut ids = Vec::new();
        let mut buf = String::new();
        let mut at = 0;
        let mut characters = 0usize;
        while at < escaped.len() {
            if characters.is_multiple_of(256) {
                poll()?;
            }
            characters += 1;
            if let Some(m) = self
                .markers
                .iter()
                .find(|m| escaped[at..].starts_with(m.as_str()))
            {
                self.bpe_controlled(&buf, &mut ids, poll)?;
                buf.clear();
                ids.push(self.lookup[m]);
                at += m.len();
            } else if let Some(c) = escaped[at..].chars().next() {
                buf.push(c);
                at += c.len_utf8();
            }
        }
        self.bpe_controlled(&buf, &mut ids, poll)?;
        poll()?;
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut buf = Vec::new();
        for &id in ids {
            buf.extend(self.token_bytes(id)?);
        }
        let text = String::from_utf8_lossy(&buf).replace('▁', " ");
        Ok(if self.dummy {
            text.strip_prefix(' ').unwrap_or(&text).to_string()
        } else {
            text
        })
    }

    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let i = id as usize;
        let piece = self
            .pieces
            .get(i)
            .ok_or_else(|| Error::Tokenizer("token id outside vocabulary".into()))?;
        Ok(match self.kinds[i] {
            4 => vec![u8::from_str_radix(&piece[3..5], 16)
                .map_err(|_| Error::Tokenizer("invalid byte piece".into()))?],
            1 | 2 => Vec::new(),
            _ => piece.replace('▁', " ").into_bytes(),
        })
    }

    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.lookup.get(piece).copied()
    }

    #[cfg(test)]
    pub(crate) fn alias_piece(&mut self, piece: &str, id: u32) {
        self.lookup.insert(piece.to_string(), id);
    }
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    pub(crate) fn max_token_bytes(&self) -> usize {
        self.pieces.iter().map(String::len).max().unwrap_or(0)
    }
}

fn infallible<T>(result: std::result::Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;
