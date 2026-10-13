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
