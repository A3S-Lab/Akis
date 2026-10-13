#![cfg(test)]
pub(crate) fn tokenizer_blob() -> Vec<u8> {
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

pub(crate) fn fixture() -> Vec<u8> {
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
    ts.push((vec![2], vec![0.0, 0.0]));
    h[1] = (ts.len() + 1) as u32;
    let mut records: Vec<(u8, Vec<usize>, Vec<u8>)> = ts
        .into_iter()
        .map(|(s, v)| (2, s, v.iter().flat_map(|v| v.to_le_bytes()).collect()))
        .collect();
    records.push((4, Vec::new(), tokenizer_blob()));
    let mut out: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();
    out.extend(vec![0; 28 * 4]);
    let start = out.len();
    out.resize(start + records.len() * 44, 0);
    for (i, (dtype, shape, bytes)) in records.into_iter().enumerate() {
        out.resize(out.len().div_ceil(64) * 64, 0);
        let offset = out.len();
        let at = start + i * 44;
        out[at] = dtype;
        out[at + 1] = shape.len() as u8;
        for (j, n) in shape.iter().enumerate() {
            out[at + 4 + j * 4..at + 8 + j * 4].copy_from_slice(&(*n as u32).to_le_bytes());
        }
        out[at + 20..at + 28].copy_from_slice(&(offset as u64).to_le_bytes());
        out[at + 28..at + 36].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend(bytes);
    }
    out
}
