pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
}

pub(crate) fn rms(x: &[f32]) -> Vec<f32> {
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-6).sqrt();
    x.iter().map(|v| v / scale).collect()
}

pub(crate) fn norm(x: &[f32], w: &[f32]) -> Vec<f32> {
    rms(x)
        .iter()
        .zip(w.iter().cycle())
        .map(|(v, w)| v * (1.0 + w))
        .collect()
}

pub(crate) fn quant(x: &mut [f32]) {
    let max = x.iter().map(|v| v.abs()).fold(0.0, f32::max);
    if max == 0.0 {
        return;
    }
    let scale = max / 127.0;
    for v in x {
        *v = (*v / scale).round_ties_even().clamp(-128.0, 127.0) * scale;
    }
}

pub(crate) fn rope(x: &mut [f32], pos: usize, theta: f32) {
    let half = x.len() / 2;
    for i in 0..half {
        let angle = pos as f32 / theta.powf(i as f32 / half as f32);
        let (s, c) = angle.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = b * c + a * s;
    }
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

pub(crate) fn sinkhorn(x: &mut [f32], n: usize) {
    for _ in 0..20 {
        for row in x.chunks_exact_mut(n) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z = max + row.iter().map(|v| (*v - max).exp()).sum::<f32>().ln();
            for v in row {
                *v -= z;
            }
        }
        for j in 0..n {
            let max = (0..n)
                .map(|i| x[i * n + j])
                .fold(f32::NEG_INFINITY, f32::max);
            let z = max + (0..n).map(|i| (x[i * n + j] - max).exp()).sum::<f32>().ln();
            for i in 0..n {
                x[i * n + j] -= z;
            }
        }
    }
    for v in x {
        *v = v.exp();
    }
}

#[cfg(test)]
#[path = "math_tests.rs"]
mod tests;
