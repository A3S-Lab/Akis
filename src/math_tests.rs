#![cfg(test)]
use super::*;
#[test]
fn stable_softmax_and_quantization() {
    let mut a = [1000.0, 1000.0];
    softmax(&mut a);
    assert_eq!(a, [0.5, 0.5]);
    let mut x = [-127.0, -0.5, 0.5, 127.0];
    quant(&mut x);
    assert_eq!(x, [-127.0, 0.0, 0.0, 127.0]);
}
#[test]
fn sinkhorn_preserves_both_marginals() {
    let mut x = [2.0, -1.0, 0.0, 3.0];
    sinkhorn(&mut x, 2);
    for v in [x[0] + x[1], x[2] + x[3], x[0] + x[2], x[1] + x[3]] {
        assert!((v - 1.0).abs() < 0.002);
    }
}
#[test]
fn split_half_rope_is_a_rotation() {
    let mut x = [1.0, 2.0, 3.0, 4.0];
    let before = dot(&x, &x);
    rope(&mut x, 7, 10000.0);
    assert!((dot(&x, &x) - before).abs() < 1e-4);
}
