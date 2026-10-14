#![cfg(test)]
use super::*;
use crate::test_support::fixture;

#[test]
fn validates_real_layout_and_rejects_every_truncation() {
    let bytes = fixture();
    let limits = InferenceLimits::default();
    let archive = Archive::from_bytes(bytes.clone(), &limits).unwrap();
    assert_eq!(archive.geometry.layers, 1);
    for n in 0..bytes.len() {
        assert!(
            Archive::from_bytes(bytes[..n].to_vec(), &limits).is_err(),
            "length {n}"
        );
    }
}

#[test]
fn rejects_overlap_unknown_format_and_non_finite_weights() {
    let original = fixture();
    let limits = InferenceLimits::default();
    for at in [0, 196 + 112 + 20, 196 + 112 + 28] {
        let mut b = original.clone();
        b[at..at + 4].fill(0xff);
        assert!(Archive::from_bytes(b, &limits).is_err());
    }
    let mut archive = Archive::from_bytes(original, &limits).unwrap();
    let at = archive.records[0].offset;
    archive.data[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(archive.values(0).is_err());
}

#[test]
fn hadamard_is_orthonormal() {
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut x = original.clone();
    hadamard(&mut x);
    hadamard(&mut x);
    for (a, b) in x.iter().zip(original) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn tokenizer_matches_known_bpe_and_marker_boundaries() {
    let tok = Tokenizer::from_bytes(&crate::test_support::tokenizer_blob(), 10).unwrap();
    assert_eq!(tok.encode("hi 你<tool_call>"), vec![7, 6, 7, 8, 9]);
    assert_eq!(tok.decode(&[7, 6, 7, 8, 9]).unwrap(), "hi 你<tool_call>");
    assert!(tok.decode(&[999]).is_err());
    assert_eq!(tok.encode(""), Vec::<u32>::new());
}

#[test]
fn directories_and_over_budget_artifacts_are_rejected() {
    assert!(Archive::open(env!("CARGO_MANIFEST_DIR"), &InferenceLimits::default()).is_err());
    let bytes = fixture();
    let limits = InferenceLimits {
        max_model_bytes: bytes.len() as u64 - 1,
        ..InferenceLimits::default()
    };
    assert!(Archive::from_bytes(bytes, &limits).is_err());
}

#[test]
fn every_cq_width_matches_an_independent_walsh_matrix() {
    for width in 1..=5 {
        let bits = if width == 5 { 2 } else { width };
        let mut archive = Archive::from_bytes(fixture(), &InferenceLimits::default()).unwrap();
        archive.codebook = (0..28).map(|i| (i as f32 - 13.0) / 20.0).collect();
        let rows = 2;
        let cols = 129;
        let padded = 256;
        let row_bytes = padded * bits / 8;
        let mut packed = vec![0u8; rows * row_bytes];
        let mut expected = Vec::new();
        for row in 0..rows {
            for group in 0..2 {
                let norm = (row * 2 + group) as f32;
                let mut centroids = Vec::new();
                for j in 0..128 {
                    let index = if width == 5 {
                        [0, 1, 3][(j + row + group) % 3]
                    } else {
                        (j * 7 + row + group) % (1 << bits)
                    };
                    let bit = (group * 128 + j) * bits;
                    for k in 0..bits {
                        packed[row * row_bytes + (bit + k) / 8] |=
                            (((index >> k) & 1) as u8) << ((bit + k) % 8);
                    }
                    let centroid = match width {
                        1 => {
                            if index == 0 {
                                -(2.0 / std::f32::consts::PI / 128.0).sqrt()
                            } else {
                                (2.0 / std::f32::consts::PI / 128.0).sqrt()
                            }
                        }
                        5 => match index {
                            0 => 0.0,
                            1 => 1.2240064 / 128f32.sqrt(),
                            _ => -1.2240064 / 128f32.sqrt(),
                        },
                        _ => archive.codebook[(1 << width) - 4 + index],
                    };
                    centroids.push(centroid * norm);
                }
                for i in 0usize..128.min(cols - group * 128) {
                    let value: f32 = centroids
                        .iter()
                        .enumerate()
                        .map(|(j, c)| {
                            if (i & j).count_ones().is_multiple_of(2) {
                                *c
                            } else {
                                -*c
                            }
                        })
                        .sum();
                    expected.push(value / 128f32.sqrt());
                }
            }
        }
        for norm in 0..4 {
            packed.extend(f16::from_f32(norm as f32).to_le_bytes());
        }
        let offset = archive.data.len();
        archive.data.extend(&packed);
        archive.records[0] = Record {
            dtype: 3,
            shape: vec![rows, cols],
            offset,
            bytes: packed.len(),
            group: 128,
            bits: width,
        };
        let actual = archive.values(0).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!((a - b).abs() < 1e-4, "CQ {width} element {i}: {a} != {b}");
        }
        let table = archive.cq_gather(0).unwrap();
        for row in 0..rows {
            assert_eq!(
                table.row(row, &archive.codebook).unwrap(),
                actual[row * cols..(row + 1) * cols]
            );
        }
        if width == 5 {
            archive.data[offset] = (archive.data[offset] & !3) | 2;
            assert!(archive.values(0).is_err());
        }
    }
}

#[test]
fn bounded_archive_mutations_never_panic_or_bypass_limits() {
    let original = fixture();
    let limits = InferenceLimits {
        max_tensor_elements: 1024,
        max_model_bytes: 65536,
        max_resident_weight_bytes: 65536,
        ..InferenceLimits::default()
    };
    let mut random = 0xa5ed91u32;
    for _ in 0..2000 {
        let mut bytes = original.clone();
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        let at = random as usize % bytes.len();
        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
        bytes[at] ^= (random >> 24) as u8;
        if let Ok(archive) = Archive::from_bytes(bytes, &limits) {
            let runtime =
                crate::EmbeddedRuntime::new(crate::DevicePreference::Cpu, limits.clone()).unwrap();
            if let Ok(model) = crate::Model::load(archive, runtime) {
                let _ = model.logits(&[2, 4], &crate::CancellationToken::new());
            }
        }
    }
}

fn patched(edits: &[(usize, u32)]) -> Result<Archive> {
    let mut bytes = fixture();
    for &(index, value) in edits {
        let at = index * 4;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    Archive::from_bytes(bytes, &InferenceLimits::default())
}

fn assert_open(edits: &[(usize, u32)]) -> Archive {
    match patched(edits) {
        Ok(archive) => archive,
        Err(error) => panic!("edits {edits:?} rejected: {error}"),
    }
}

fn assert_closed(edits: &[(usize, u32)], message: &str) {
    match patched(edits) {
        Ok(_) => panic!("edits {edits:?} opened"),
        Err(error) => assert!(
            error.to_string().contains(message),
            "edits {edits:?}: {error}"
        ),
    }
}

#[test]
fn geometry_edges_accept_the_limit_and_reject_one_past_it() {
    let stock = Archive::from_bytes(fixture(), &InferenceLimits::default()).unwrap();
    assert_eq!(stock.geometry().layers, 1);
    assert_eq!(stock.geometry().rope_theta, 10_000.0);
    let geometry = "invalid model geometry";
    let engram = "invalid engram geometry";
    let counts = "invalid engram counts";

    for index in [5, 7, 8, 9, 10, 11, 12, 13, 15] {
        assert_closed(&[(index, 0)], geometry);
    }
    assert_eq!(assert_open(&[(10, 64)]).geometry().layers, 64);
    assert_closed(&[(10, 65)], geometry);
    for index in [8, 15, 19, 22, 23, 24, 25] {
        let archive = assert_open(&[(index, 256)]);
        let value = match index {
            8 => archive.geometry().heads,
            15 => archive.geometry().lanes,
            19 => archive.geometry().conv_taps,
            22 => archive.geometry().engram_tables,
            23 => archive.geometry().engram_taps,
            24 => archive.geometry().engram_dilation,
            25 => archive.geometry().engram_seed_heads,
            _ => unreachable!(),
        };
        assert_eq!(value, 256);
        assert_closed(&[(index, 257)], geometry);
    }
    assert_closed(&[(9, 256)], geometry);
    assert_eq!(assert_open(&[(8, 256), (9, 256)]).geometry().kv_heads, 256);
    assert_closed(&[(9, 257)], geometry);

    let cap = 1_048_576u32;
    for index in [5, 11, 12, 13, 20, 21] {
        let _ = assert_open(&[(index, cap)]);
        assert_closed(&[(index, cap + 1)], geometry);
    }
    let wide = assert_open(&[(7, cap), (14, cap)]);
    assert_eq!(wide.geometry().width, cap as usize);
    assert_eq!(wide.geometry().hadamard, cap as usize);
    assert_closed(&[(7, cap + 1), (14, cap)], geometry);
    assert_eq!(assert_open(&[(14, cap)]).geometry().hadamard, cap as usize);
    assert_closed(&[(14, cap + 1)], geometry);

    assert_closed(&[(11, 1)], geometry);
    assert_closed(&[(11, 3)], geometry);
    assert_eq!(assert_open(&[(11, 4)]).geometry().qk_dim, 4);
    assert_closed(&[(8, 2), (9, 3)], geometry);
    assert_eq!(assert_open(&[(8, 2), (9, 2)]).geometry().kv_heads, 2);
    assert_eq!(assert_open(&[(6, 10)]).geometry().out_vocab, 10);
    assert_eq!(assert_open(&[(6, 0)]).geometry().out_vocab, 10);
    assert_closed(&[(6, 11)], geometry);
    assert_closed(&[(14, 3)], geometry);
    assert_closed(&[(14, 1)], geometry);
    assert_eq!(assert_open(&[(14, 4)]).geometry().hadamard, 4);

    for theta in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_closed(&[(48, theta.to_bits())], geometry);
    }
    assert_eq!(
        assert_open(&[(48, f32::MIN_POSITIVE.to_bits())])
            .geometry()
            .rope_theta,
        f32::MIN_POSITIVE
    );

    assert_closed(&[(26, 5)], counts);
    assert_closed(&[(31, 17)], counts);
    assert_eq!(
        assert_open(&[(26, 4), (27, 1), (28, 2), (29, 3), (30, 4)])
            .geometry()
            .engram_orders,
        vec![1, 2, 3, 4]
    );
    assert_closed(&[(26, 1), (27, 257)], geometry);
    assert_eq!(
        assert_open(&[(26, 1), (27, 256)]).geometry().engram_orders,
        vec![256]
    );
    assert_eq!(
        assert_open(&[(26, 1), (27, 0)]).geometry().engram_orders,
        vec![0]
    );

    let ready = [
        (20, 1),
        (21, 1),
        (22, 1),
        (23, 1),
        (24, 1),
        (26, 1),
        (27, 2),
    ];
    let mut one_site = vec![(10, 4), (31, 1), (32, 3)];
    one_site.extend(ready);
    let opened = assert_open(&one_site);
    assert_eq!(opened.geometry().layers, 4);
    assert_eq!(opened.geometry().engram_sites, vec![3]);
    assert_closed(&[(10, 4), (31, 1), (32, 4)], geometry);
    assert_closed(&[(10, 4), (31, 2), (32, 2), (33, 1)], geometry);
    assert_closed(&[(10, 4), (31, 2), (32, 1), (33, 1)], geometry);

    let mut full_sites = vec![(10, 16), (31, 16)];
    full_sites.extend(ready);
    for site in 0..16u32 {
        full_sites.push((32 + site as usize, site));
    }
    assert_eq!(
        assert_open(&full_sites).geometry().engram_sites,
        (0..16).collect::<Vec<_>>()
    );

    let mut broken = vec![(31, 1), (32, 0), (26, 1), (27, 0)];
    broken.extend([(20, 1), (21, 1), (22, 1), (23, 1), (24, 1)]);
    assert_closed(&broken, engram);
    assert_closed(
        &[
            (31, 1),
            (32, 0),
            (20, 0),
            (21, 1),
            (22, 1),
            (23, 1),
            (24, 1),
            (26, 1),
            (27, 2),
        ],
        engram,
    );
    assert_closed(
        &[
            (31, 1),
            (32, 0),
            (20, 1),
            (21, 1),
            (22, 1),
            (23, 1),
            (24, 1),
            (26, 2),
            (27, 2),
            (28, 3),
        ],
        engram,
    );
    assert_eq!(
        assert_open(&[
            (31, 1),
            (32, 0),
            (20, 1),
            (21, 1),
            (22, 2),
            (23, 1),
            (24, 1),
            (26, 2),
            (27, 2),
            (28, 3),
        ])
        .geometry()
        .engram_tables,
        2
    );
}

fn record_slot(index: usize) -> usize {
    HEADER + 28 * 4 + index * RECORD
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn parse_archive(bytes: Vec<u8>) -> Result<Archive> {
    Archive::from_bytes(bytes, &InferenceLimits::default())
}

fn reject_record(bytes: Vec<u8>, message: &str) {
    match parse_archive(bytes) {
        Ok(_) => panic!("archive opened, expected {message}"),
        Err(error) => assert!(error.to_string().contains(message), "{error}"),
    }
}

fn reject_record_without(bytes: Vec<u8>, message: &str, absent: &str) {
    match parse_archive(bytes) {
        Ok(_) => panic!("archive opened, expected {message}"),
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains(message), "{text}");
            assert!(!text.contains(absent), "{text}");
        }
    }
}

#[test]
fn tensor_records_accept_a_matching_length_and_reject_one_bad_field() {
    let bytes = fixture();
    let stock = parse_archive(bytes.clone()).unwrap();
    assert!(stock.records.len() > 1);
    let first = &stock.records[0];
    let last_index = stock.records.len() - 1;
    let last = &stock.records[last_index];
    assert_eq!(first.dtype, 2);
    assert_eq!(first.shape, [10, 2]);
    assert_eq!(first.offset % 64, 0);
    assert!(first.bytes > 1);
    assert!(first.offset + first.bytes < bytes.len());
    assert_eq!(last.dtype, 4);
    assert!(last.shape.is_empty());
    assert_eq!(last.offset + last.bytes, bytes.len());
    assert!(stock.records[..last_index]
        .iter()
        .all(|record| record.dtype != 4));
    let slot = record_slot(0);
    let last_slot = record_slot(last_index);
    let invalid = "invalid tensor record";
    let mismatch = "tensor 0 byte length mismatch";
    let range = "invalid or overlapping tensor range";

    for dtype in [0_u8, 5, 255] {
        let mut bad = bytes.clone();
        bad[slot] = dtype;
        reject_record(bad, invalid);
    }
    let mut half = bytes.clone();
    half[slot] = 1;
    reject_record_without(half, mismatch, invalid);

    let mut wide = bytes.clone();
    wide[slot + 1] = 5;
    reject_record(wide, invalid);
    let mut reserved = bytes.clone();
    reserved[slot + 2] = 1;
    reject_record(reserved, invalid);

    let mut four = bytes.clone();
    four[slot + 1] = 4;
    put_u32(&mut four, slot + 12, 1);
    put_u32(&mut four, slot + 16, 1);
    let opened = parse_archive(four).unwrap();
    assert_eq!(opened.records[0].dtype, 2);
    assert_eq!(opened.records[0].shape, [10, 2, 1, 1]);
    assert_eq!(opened.records[0].bytes, first.bytes);

    let mut four_long = bytes.clone();
    four_long[slot + 1] = 4;
    put_u32(&mut four_long, slot + 12, 1);
    put_u32(&mut four_long, slot + 16, 2);
    reject_record_without(four_long, mismatch, invalid);

    let mut empty = bytes.clone();
    empty[slot + 1] = 0;
    reject_record_without(empty, "positive dimensions", invalid);
    let mut zero_axis = bytes.clone();
    zero_axis[slot + 1] = 3;
    reject_record_without(zero_axis, "positive dimensions", "byte length mismatch");

    let mut huge = bytes.clone();
    huge[slot + 1] = 3;
    put_u32(&mut huge, slot + 4, u32::MAX);
    put_u32(&mut huge, slot + 8, u32::MAX);
    put_u32(&mut huge, slot + 12, 2);
    reject_record_without(huge, "dimensions overflowed", "byte length mismatch");

    let cq = "unsupported CQ geometry";
    let mut cq_stock = bytes.clone();
    cq_stock[slot] = 3;
    reject_record_without(cq_stock, cq, invalid);
    let mut cq_bits = bytes.clone();
    cq_bits[slot] = 3;
    put_u32(&mut cq_bits, slot + 36, 128);
    put_u32(&mut cq_bits, slot + 40, 6);
    reject_record(cq_bits, cq);
    let mut cq_rank = bytes.clone();
    cq_rank[slot] = 3;
    cq_rank[slot + 1] = 1;
    put_u32(&mut cq_rank, slot + 36, 128);
    put_u32(&mut cq_rank, slot + 40, 1);
    reject_record(cq_rank, cq);
    for bits in [1_u32, 5] {
        let mut cq_len = bytes.clone();
        cq_len[slot] = 3;
        put_u32(&mut cq_len, slot + 36, 128);
        put_u32(&mut cq_len, slot + 40, bits);
        reject_record_without(cq_len, mismatch, cq);
    }

    let mut raw_matrix = bytes.clone();
    raw_matrix[slot] = 4;
    reject_record(raw_matrix, "RAW attachment must be scalar");
    let mut second_raw = bytes.clone();
    second_raw[slot] = 4;
    second_raw[slot + 1] = 0;
    reject_record(second_raw, "expected one trailing tokenizer");
    assert_eq!(last.bytes % 4, 0);
    let mut numeric_tail = bytes.clone();
    numeric_tail[last_slot] = 2;
    numeric_tail[last_slot + 1] = 1;
    put_u32(
        &mut numeric_tail,
        last_slot + 4,
        u32::try_from(last.bytes / 4).unwrap(),
    );
    reject_record(numeric_tail, "expected one trailing tokenizer");

    let mut short = bytes.clone();
    put_u64(&mut short, slot + 28, (first.bytes - 1) as u64);
    reject_record_without(short, mismatch, "overlapping");
    let mut long = bytes.clone();
    put_u64(&mut long, slot + 28, (first.bytes + 1) as u64);
    reject_record_without(long, mismatch, "overlapping");
    let mut zero_len = bytes.clone();
    put_u64(&mut zero_len, slot + 28, 0);
    reject_record_without(zero_len, range, "byte length mismatch");
    let mut unaligned = bytes.clone();
    put_u64(&mut unaligned, slot + 20, (first.offset + 1) as u64);
    reject_record_without(unaligned, range, "byte length mismatch");
    let mut inside = bytes.clone();
    put_u64(&mut inside, slot + 20, 0);
    reject_record_without(inside, range, "byte length mismatch");
    let mut past = bytes.clone();
    put_u64(
        &mut past,
        slot + 28,
        (bytes.len() - first.offset + 1) as u64,
    );
    reject_record_without(past, range, "byte length mismatch");
    let mut tail_past = bytes.clone();
    put_u64(&mut tail_past, last_slot + 28, (last.bytes + 1) as u64);
    reject_record_without(tail_past, range, "byte length mismatch");
    let mut overflow = bytes.clone();
    put_u64(
        &mut overflow,
        slot + 28,
        u64::try_from(usize::MAX - first.offset).unwrap() + 1,
    );
    reject_record_without(overflow, "tensor range overflow", "overlapping");

    let mut tail_short = bytes.clone();
    put_u64(&mut tail_short, last_slot + 28, (last.bytes - 1) as u64);
    let shortened = parse_archive(tail_short).unwrap();
    assert_eq!(shortened.records[last_index].dtype, 4);
    assert_eq!(shortened.records[last_index].bytes, last.bytes - 1);
    assert!(shortened.records[..last_index]
        .iter()
        .all(|record| record.dtype != 4));
    match shortened.tokenizer() {
        Ok(_) => panic!("truncated tokenizer parsed"),
        Err(error) => assert!(
            error.to_string().contains("malformed SentencePiece"),
            "{error}"
        ),
    }

    let directory = HEADER + 28 * 4;
    assert!(directory + 1_000 * RECORD > bytes.len());
    let mut truncated = bytes;
    put_u32(&mut truncated, 4, 1_000);
    reject_record_without(truncated, "truncated directory", invalid);
}

fn load_cpu(archive: Archive) -> Result<crate::Model> {
    crate::Model::load(
        archive,
        crate::EmbeddedRuntime::new(crate::DevicePreference::Cpu, InferenceLimits::default())
            .unwrap(),
    )
}

fn reject_load(archive: Archive, message: &str) {
    match load_cpu(archive) {
        Ok(_) => panic!("model loaded, expected {message}"),
        Err(error) => assert!(error.to_string().contains(message), "{error}"),
    }
}

fn reject_load_without(archive: Archive, message: &str, absent: &str) {
    match load_cpu(archive) {
        Ok(_) => panic!("model loaded, expected {message}"),
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains(message), "{text}");
            assert!(!text.contains(absent), "{text}");
        }
    }
}

fn permutation_offsets(archive: &Archive) -> Vec<usize> {
    archive
        .records
        .iter()
        .filter(|record| record.dtype == 2 && record.shape == [2] && record.bytes == 8)
        .filter(|record| {
            let data = &archive.data[record.offset..record.offset + 8];
            f32::from_le_bytes(data[0..4].try_into().unwrap()) == 0.0
                && f32::from_le_bytes(data[4..8].try_into().unwrap()) == 1.0
        })
        .map(|record| record.offset)
        .collect()
}

fn write_f32s(bytes: &mut [u8], offset: usize, values: &[f32]) {
    for (index, value) in values.iter().enumerate() {
        let at = offset + index * 4;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn cq_payload(stored_bits: usize, norm: f32, crumb: u8) -> Vec<u8> {
    let width = if stored_bits == 5 { 2 } else { stored_bits };
    let row_bytes = 128 * width / 8;
    let mut payload = vec![0u8; row_bytes + 2];
    payload[0] = crumb;
    payload[row_bytes..row_bytes + 2].copy_from_slice(&half::f16::from_f32(norm).to_le_bytes());
    payload
}

fn install_cq(archive: &mut Archive, stored_bits: usize, payload: Vec<u8>) {
    let offset = archive.data.len();
    let bytes = payload.len();
    archive.data.extend(payload);
    archive.records[0] = Record {
        dtype: 3,
        shape: vec![1, 1],
        offset,
        bytes,
        group: 128,
        bits: stored_bits,
    };
}

#[test]
fn numeric_payloads_and_load_gates_fail_closed() {
    let stock_bytes = fixture();
    let stock = parse_archive(stock_bytes.clone()).unwrap();
    let embedding = stock.values(0).unwrap();
    assert_eq!(embedding.len(), 20);
    assert!(embedding.iter().all(|value| value.is_finite()));
    let loaded = load_cpu(parse_archive(stock_bytes.clone()).unwrap()).unwrap();
    assert_eq!(loaded.geometry().kv_bits, 8);
    let last = stock.records.len() - 1;
    assert_eq!(stock.records[last].dtype, 4);
    match stock.values(last) {
        Ok(_) => panic!("raw tokenizer decoded as a tensor"),
        Err(error) => assert!(
            error.to_string().contains("RAW is not a numeric tensor"),
            "{error}"
        ),
    }
    match stock.values(stock.records.len()) {
        Ok(_) => panic!("missing tensor decoded"),
        Err(error) => assert!(error.to_string().contains("missing tensor"), "{error}"),
    }
    match stock.cq_gather(0) {
        Ok(_) => panic!("f32 record opened as a gather table"),
        Err(error) => assert!(
            error
                .to_string()
                .contains("gather table must be a CQ matrix"),
            "{error}"
        ),
    }

    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = stock_bytes.clone();
        write_f32s(&mut bytes, stock.records[0].offset, &[value]);
        let archive = parse_archive(bytes).unwrap();
        match archive.values(0) {
            Ok(_) => panic!("non-finite embedding decoded"),
            Err(error) => assert!(error.to_string().contains("non-finite tensor"), "{error}"),
        }
    }
    let mut negative_zero = stock_bytes.clone();
    write_f32s(
        &mut negative_zero,
        stock.records[0].offset,
        &[f32::from_bits(0x8000_0000)],
    );
    let archive = parse_archive(negative_zero).unwrap();
    assert_eq!(archive.values(0).unwrap()[0].to_bits(), 0x8000_0000);
    load_cpu(archive).unwrap();

    let mut half_bytes = stock_bytes.clone();
    let slot = record_slot(0);
    half_bytes[slot] = 1;
    put_u64(&mut half_bytes, slot + 28, 40);
    let finite_half = half::f16::from_f32(1.0).to_le_bytes();
    for index in 0..20 {
        let at = stock.records[0].offset + index * 2;
        half_bytes[at..at + 2].copy_from_slice(&finite_half);
    }
    let half_archive = parse_archive(half_bytes.clone()).unwrap();
    assert_eq!(half_archive.records[0].dtype, 1);
    assert_eq!(half_archive.values(0).unwrap(), vec![1.0; 20]);
    load_cpu(parse_archive(half_bytes).unwrap()).unwrap();
    let mut half_nan = stock_bytes.clone();
    half_nan[slot] = 1;
    put_u64(&mut half_nan, slot + 28, 40);
    for index in 0..20 {
        let at = stock.records[0].offset + index * 2;
        half_nan[at..at + 2].copy_from_slice(&finite_half);
    }
    half_nan[stock.records[0].offset..stock.records[0].offset + 2]
        .copy_from_slice(&half::f16::from_f32(f32::NAN).to_le_bytes());
    match parse_archive(half_nan).unwrap().values(0) {
        Ok(_) => panic!("non-finite f16 embedding decoded"),
        Err(error) => assert!(error.to_string().contains("non-finite tensor"), "{error}"),
    }

    for value in [f32::NAN, f32::INFINITY] {
        let mut bytes = stock_bytes.clone();
        bytes[HEADER..HEADER + 4].copy_from_slice(&value.to_le_bytes());
        reject_record_without(bytes, "non-finite codebook", "invalid tensor record");
    }
    for value in [1.0, f32::from_bits(0x8000_0000)] {
        let mut bytes = stock_bytes.clone();
        bytes[HEADER..HEADER + 4].copy_from_slice(&value.to_le_bytes());
        let archive = parse_archive(bytes).unwrap();
        assert_eq!(archive.codebook()[0].to_bits(), value.to_bits());
        load_cpu(archive).unwrap();
    }

    for bits in [0_u32, 7, 9] {
        let mut bytes = stock_bytes.clone();
        put_u32(&mut bytes, 16, bits);
        let archive = parse_archive(bytes).unwrap();
        assert_eq!(archive.geometry().kv_bits, bits as usize);
        reject_load(archive, "only A8 KV archives are supported");
    }

    let mut short = parse_archive(stock_bytes.clone()).unwrap();
    short.records.pop();
    reject_load_without(short, "missing model tensors", "only A8 KV");

    assert_eq!(stock.records[2].shape, [2, 2]);
    let mut rank = parse_archive(stock_bytes.clone()).unwrap();
    rank.records[2].shape = vec![2];
    reject_load_without(rank, "projection rank mismatch", "non-finite");

    let permutations = permutation_offsets(&stock);
    assert_eq!(permutations.len(), 2);
    let mut swapped = stock_bytes.clone();
    write_f32s(&mut swapped, permutations[0], &[1.0, 0.0]);
    load_cpu(parse_archive(swapped).unwrap()).unwrap();
    for values in [[0.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.5, 1.0]] {
        let mut bytes = stock_bytes.clone();
        write_f32s(&mut bytes, permutations[0], &values);
        reject_load_without(
            parse_archive(bytes).unwrap(),
            "invalid Hadamard permutation",
            "non-finite",
        );
    }
    let mut second = stock_bytes.clone();
    write_f32s(&mut second, permutations[1], &[0.0, 0.0]);
    reject_load(
        parse_archive(second).unwrap(),
        "invalid Hadamard permutation",
    );
    let mut poisoned = stock_bytes.clone();
    write_f32s(&mut poisoned, permutations[1], &[f32::NAN, 1.0]);
    reject_load_without(
        parse_archive(poisoned).unwrap(),
        "non-finite tensor",
        "invalid Hadamard permutation",
    );

    let mut finite = parse_archive(stock_bytes.clone()).unwrap();
    install_cq(&mut finite, 1, cq_payload(1, 1.0, 0));
    let row = finite.values(0).unwrap();
    assert_eq!(row.len(), 1);
    assert!(row[0].is_finite());
    let table = finite.cq_gather(0).unwrap();
    assert_eq!(table.row(0, finite.codebook()).unwrap(), row);
    match table.row(1, finite.codebook()) {
        Ok(_) => panic!("cq row 1 decoded"),
        Err(error) => assert!(error.to_string().contains("CQ row out of range"), "{error}"),
    }

    let mut short_row = parse_archive(stock_bytes.clone()).unwrap();
    install_cq(&mut short_row, 1, vec![0; 16]);
    match short_row.values(0) {
        Ok(_) => panic!("short cq row decoded"),
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains("truncated CQ norm"), "{text}");
            assert!(!text.contains("truncated CQ row"), "{text}");
        }
    }

    let mut wide_bits = parse_archive(stock_bytes).unwrap();
    install_cq(&mut wide_bits, 17, vec![0; 4]);
    match wide_bits.values(0) {
        Ok(_) => panic!("17-bit cq row decoded"),
        Err(error) => assert!(
            error.to_string().contains("unsupported CQ geometry"),
            "{error}"
        ),
    }
    let mut past_codebook = parse_archive(fixture()).unwrap();
    install_cq(&mut past_codebook, 6, cq_payload(6, 0.0, 0));
    match past_codebook.values(0) {
        Ok(_) => panic!("cq codebook index 60 decoded"),
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains("CQ codebook index out of range"), "{text}");
            assert!(!text.contains("unsupported CQ geometry"), "{text}");
        }
    }
    let mut covered = parse_archive(fixture()).unwrap();
    install_cq(&mut covered, 4, cq_payload(4, 0.0, 0));
    assert!(covered
        .values(0)
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));

    for crumb in [0_u8, 1, 3] {
        let mut ternary = parse_archive(fixture()).unwrap();
        install_cq(&mut ternary, 5, cq_payload(5, 1.0, crumb));
        assert!(ternary
            .values(0)
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
    let mut crumb = parse_archive(fixture()).unwrap();
    install_cq(&mut crumb, 5, cq_payload(5, 1.0, 2));
    match crumb.values(0) {
        Ok(_) => panic!("ternary crumb 2 decoded"),
        Err(error) => assert!(
            error.to_string().contains("invalid ternary crumb"),
            "{error}"
        ),
    }

    let mut nan_norm = parse_archive(fixture()).unwrap();
    install_cq(&mut nan_norm, 1, cq_payload(1, f32::NAN, 0));
    match nan_norm.values(0) {
        Ok(_) => panic!("non-finite cq norm decoded"),
        Err(error) => assert!(error.to_string().contains("non-finite tensor"), "{error}"),
    }
    match nan_norm.cq_gather(0) {
        Ok(_) => panic!("non-finite cq gather opened"),
        Err(error) => assert!(error.to_string().contains("non-finite tensor"), "{error}"),
    }

    let mut no_group = parse_archive(fixture()).unwrap();
    no_group.records[0].dtype = 3;
    no_group.records[0].group = 0;
    no_group.records[0].bits = 1;
    no_group.records[0].shape = vec![1, 1];
    match no_group.values(0) {
        Ok(_) => panic!("zero-group row decoded"),
        Err(error) => assert!(error.to_string().contains("CQ row out of range"), "{error}"),
    }
    match no_group.cq_gather(0) {
        Ok(_) => panic!("zero-group gather opened"),
        Err(error) => assert!(
            error
                .to_string()
                .contains("gather table must be a CQ matrix"),
            "{error}"
        ),
    }
}

fn final_norm_index(archive: &Archive) -> usize {
    let geometry = archive.geometry();
    let stride = if geometry.conv_taps > 0 { 27 } else { 24 };
    let engrams = 1 + stride * geometry.layers + 11;
    engrams + 4 * geometry.engram_sites.len()
}

fn push_numeric(archive: &mut Archive, shape: &[usize], values: &[f32]) {
    assert_eq!(values.len(), shape.iter().copied().product::<usize>());
    let offset = archive.data.len();
    let bytes = values.len() * 4;
    for value in values {
        archive.data.extend(value.to_le_bytes());
    }
    let record = Record {
        dtype: 2,
        shape: shape.to_vec(),
        offset,
        bytes,
        group: 0,
        bits: 0,
    };
    let tail = archive.records.pop().expect("tokenizer");
    archive.records.push(record);
    archive.records.push(tail);
}

fn push_probe(archive: &mut Archive, code: f32, gain_rows: usize, out: usize) {
    let width = archive.geometry().width;
    let queries = 1;
    push_numeric(archive, &[1], &[code]);
    push_numeric(archive, &[gain_rows, width], &vec![0.0; gain_rows * width]);
    push_numeric(archive, &[gain_rows, 1], &vec![0.0; gain_rows]);
    push_numeric(archive, &[queries, width], &vec![0.0; queries * width]);
    push_numeric(
        archive,
        &[queries, gain_rows, 1],
        &vec![0.0; queries * gain_rows],
    );
    push_numeric(
        archive,
        &[out, queries * width],
        &vec![0.0; out * queries * width],
    );
    push_numeric(archive, &[out], &vec![0.0; out]);
    if code == 3.0 {
        push_numeric(archive, &[3], &[0.0; 3]);
    }
}

#[test]
fn probe_manifest_and_tensor_shapes_fail_closed() {
    let stock = parse_archive(fixture()).unwrap();
    let final_norm = final_norm_index(&stock);
    assert!(stock.geometry().engram_sites.is_empty());
    assert_eq!(stock.records.len(), final_norm + 2);
    assert_eq!(stock.records[1].shape, [2]);
    assert_eq!(stock.records[2].shape, [2, 2]);
    load_cpu(parse_archive(fixture()).unwrap()).unwrap();

    let mut narrow = parse_archive(fixture()).unwrap();
    narrow.records[1].shape = vec![1];
    narrow.records[1].bytes = 4;
    reject_load_without(narrow, "tensor 1 expected shape [2]", "projection rank");

    let mut wide = parse_archive(fixture()).unwrap();
    wide.records[2].shape = vec![4, 1];
    reject_load_without(wide, "tensor 2 expected shape [2, 2]", "projection rank");

    for code in [0.0, 1.5, 4.0] {
        let mut archive = parse_archive(fixture()).unwrap();
        push_numeric(&mut archive, &[1], &[code]);
        reject_load_without(
            archive,
            "invalid probe head manifest",
            "probe tensor count mismatch",
        );
    }
    let mut reversed = parse_archive(fixture()).unwrap();
    push_numeric(&mut reversed, &[2], &[2.0, 1.0]);
    reject_load_without(
        reversed,
        "invalid probe head manifest",
        "probe tensor count mismatch",
    );

    let mut counted = parse_archive(fixture()).unwrap();
    push_numeric(&mut counted, &[1], &[1.0]);
    reject_load_without(
        counted,
        "probe tensor count mismatch",
        "invalid probe head manifest",
    );
    let mut ordered = parse_archive(fixture()).unwrap();
    push_numeric(&mut ordered, &[2], &[1.0, 2.0]);
    reject_load_without(
        ordered,
        "probe tensor count mismatch",
        "invalid probe head manifest",
    );

    for (code, out) in [(1.0, 1), (2.0, 1), (3.0, 3)] {
        let mut archive = parse_archive(fixture()).unwrap();
        push_probe(&mut archive, code, 2, out);
        load_cpu(archive).unwrap();
    }

    let mut rows = parse_archive(fixture()).unwrap();
    push_probe(&mut rows, 1.0, 3, 1);
    reject_load_without(rows, "invalid probe geometry", "invalid probe output width");

    let mut confidence = parse_archive(fixture()).unwrap();
    push_probe(&mut confidence, 2.0, 2, 2);
    reject_load_without(
        confidence,
        "invalid probe output width",
        "invalid probe geometry",
    );
    let mut ternary = parse_archive(fixture()).unwrap();
    push_probe(&mut ternary, 3.0, 2, 1);
    reject_load_without(
        ternary,
        "invalid probe output width",
        "invalid probe geometry",
    );
}

fn reject_projection_length(archive: Archive) {
    match load_cpu(archive) {
        Ok(_) => panic!("projection loaded with a payload that does not match its shape"),
        Err(error) => {
            let text = error.to_string();
            assert!(text.contains("projection shape mismatch"), "{text}");
            assert!(!text.contains("expected shape"), "{text}");
            assert!(!text.contains("projection rank"), "{text}");
        }
    }
}

#[test]
fn projection_payload_length_must_match_the_shape() {
    let stock = parse_archive(fixture()).unwrap();
    assert_eq!(stock.records[0].dtype, 2);
    assert_eq!(stock.records[0].shape, [10, 2]);
    assert_eq!(stock.records[0].bytes, 80);
    assert_eq!(stock.records[1].dtype, 2);
    assert_eq!(stock.records[1].shape, [2]);
    assert_eq!(stock.records[1].bytes, 8);
    assert_eq!(stock.records[2].dtype, 2);
    assert_eq!(stock.records[2].shape, [2, 2]);
    assert_eq!(stock.records[2].bytes, 16);
    assert!(stock.records[2].offset + 20 <= stock.data.len());
    load_cpu(parse_archive(fixture()).unwrap()).unwrap();

    for bytes in [0, 12, 15, 20] {
        let mut archive = parse_archive(fixture()).unwrap();
        archive.records[2].bytes = bytes;
        reject_projection_length(archive);
    }

    let mut embedding = parse_archive(fixture()).unwrap();
    embedding.records[0].bytes = 76;
    reject_projection_length(embedding);

    let mut width = parse_archive(fixture()).unwrap();
    width.records[1].bytes = 4;
    let loaded = load_cpu(width).unwrap();
    assert_eq!(loaded.geometry().width, 2);
}

#[test]
fn permutation_length_must_match_hadamard() {
    let stock = load_cpu(parse_archive(fixture()).unwrap()).unwrap();
    assert_eq!(stock.geometry().hadamard, 2);

    let opened = assert_open(&[(14, 4)]);
    assert_eq!(opened.geometry().hadamard, 4);
    assert_eq!(permutation_offsets(&opened).len(), 2);
    match load_cpu(opened) {
        Ok(_) => panic!("hadamard 4 loaded with a length-2 permutation"),
        Err(error) => {
            let text = error.to_string();
            println!("observed rejection: {text}");
            assert!(text.contains("permutation shape mismatch"), "{text}");
            assert!(!text.contains("invalid Hadamard permutation"), "{text}");
            assert!(!text.contains("expected shape"), "{text}");
            assert!(!text.contains("invalid model geometry"), "{text}");
        }
    }
}
