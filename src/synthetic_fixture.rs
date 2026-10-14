pub fn tokenizer_blob() -> Vec<u8> {
    let pieces = [
        ("<pad>", 2),
        ("</s>", 2),
        ("<s>", 2),
        ("<unk>", 1),
        ("h", 0),
        ("i", 0),
        ("hi", 0),
        ("▁", 0),
        ("你", 0),
        ("<tool_call>", 3),
    ];
    let mut b = Vec::new();
    for x in [pieces.len() as u32, 0, 1, 2, 3] {
        b.extend(x.to_le_bytes());
    }
    b.extend([1, 0, 0, 0]);
    for (i, (piece, kind)) in pieces.iter().enumerate() {
        b.extend((i as f32).to_le_bytes());
        b.push(*kind);
        b.extend((piece.len() as u16).to_le_bytes());
        b.extend(piece.as_bytes());
    }
    b
}

pub fn fixture() -> Vec<u8> {
    build(false)
}

/// One engram site whose gather table is a 1×1 CQ matrix. Group padding makes
/// the stored bytes longer than the expanded F32 matrix.
///
/// CLI process tests compile this file on its own and do not call it.
#[allow(dead_code)]
pub fn fixture_with_one_packed_gather() -> Vec<u8> {
    build(true)
}

struct Stored {
    dtype: u8,
    shape: Vec<usize>,
    bytes: Vec<u8>,
    group: u32,
    bits: u32,
}

fn build(packed_gather: bool) -> Vec<u8> {
    let mut h = [0u32; 49];
    h[0] = 0x05e12a84;
    h[2] = 28;
    h[4] = 8;
    h[5] = 10;
    h[7] = 2;
    h[8] = 1;
    h[9] = 1;
    h[10] = 1;
    h[11] = 2;
    h[12] = 2;
    h[13] = 64;
    h[14] = 2;
    h[15] = 1;
    h[48] = 10000f32.to_bits();
    if packed_gather {
        h[20] = 1;
        h[21] = 1;
        h[22] = 1;
        h[23] = 1;
        h[24] = 1;
        h[26] = 1;
        h[27] = 2;
        h[31] = 1;
        h[32] = 0;
    }
    let mut ts: Vec<(Vec<usize>, Vec<f32>)> = vec![(
        vec![10, 2],
        (0..20)
            .map(|i| if i % 3 == 0 { 0.1 } else { 0.0 })
            .collect(),
    )];
    let shapes = [
        vec![2],
        vec![2, 2],
        vec![2, 2],
        vec![2, 2],
        vec![2],
        vec![2],
        vec![2, 2],
        vec![2, 2],
        vec![2],
        vec![1],
        vec![2],
        vec![2],
        vec![2],
        vec![2],
        vec![2],
        vec![2],
        vec![1, 1],
        vec![2, 2],
        vec![1, 1],
        vec![2, 2],
        vec![1, 1],
        vec![2, 2],
        vec![2, 8],
        vec![8, 2],
    ];
    for shape in shapes {
        ts.push((shape.clone(), vec![0.0; shape.iter().product()]));
    }
    for shape in [
        vec![1],
        vec![1],
        vec![1],
        vec![1, 1],
        vec![1, 1],
        vec![1, 1, 1],
        vec![1, 2],
        vec![1, 2],
        vec![1, 2],
    ] {
        ts.push((shape.clone(), vec![0.0; shape.iter().product()]));
    }
    ts.push((vec![2], vec![0.0, 1.0]));
    ts.push((vec![2], vec![0.0, 1.0]));
    if packed_gather {
        // Site tensors sit between the permutations and the final norm.
        // The gather table is replaced with CQ bytes below.
        ts.push((vec![1, 1], vec![0.0]));
        ts.push((vec![2, 1], vec![0.0; 2]));
        ts.push((vec![2, 1], vec![0.0; 2]));
        ts.push((vec![1, 2], vec![0.0; 2]));
    }
    ts.push((vec![2], vec![0.0, 0.0]));
    h[1] = (ts.len() + 1) as u32;
    let mut records: Vec<Stored> = ts
        .into_iter()
        .map(|(shape, values)| Stored {
            dtype: 2,
            shape,
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            group: 0,
            bits: 0,
        })
        .collect();
    if packed_gather {
        // bits=1, group=128. A 1-wide row pads to 128 and stores 18 bytes.
        let gather = records.len() - 5;
        records[gather].dtype = 3;
        records[gather].bytes = vec![0; 18];
        records[gather].group = 128;
        records[gather].bits = 1;
    }
    records.push(Stored {
        dtype: 4,
        shape: Vec::new(),
        bytes: tokenizer_blob(),
        group: 0,
        bits: 0,
    });
    let mut out: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.extend(vec![0; 28 * 4]);
    let start = out.len();
    out.resize(start + records.len() * 44, 0);
    for (i, record) in records.into_iter().enumerate() {
        out.resize(out.len().div_ceil(64) * 64, 0);
        let offset = out.len();
        let at = start + i * 44;
        out[at] = record.dtype;
        out[at + 1] = record.shape.len() as u8;
        for (j, n) in record.shape.iter().enumerate() {
            out[at + 4 + j * 4..at + 8 + j * 4].copy_from_slice(&(*n as u32).to_le_bytes());
        }
        out[at + 20..at + 28].copy_from_slice(&(offset as u64).to_le_bytes());
        out[at + 28..at + 36].copy_from_slice(&(record.bytes.len() as u64).to_le_bytes());
        out[at + 36..at + 40].copy_from_slice(&record.group.to_le_bytes());
        out[at + 40..at + 44].copy_from_slice(&record.bits.to_le_bytes());
        out.extend(record.bytes);
    }
    out
}
