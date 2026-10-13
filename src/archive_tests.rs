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
