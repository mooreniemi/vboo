#![allow(unused)]
/// This is just a "sandbox" to play with simd in.
use packed_simd::f32x4;
use rayon::prelude::*;

/// sequential, from packed_simd with adjustment to handle non-4 sized
fn simd_sum(x: &[f32]) -> f32 {
    let closest_under = x.len() - (x.len() % 4);
    let mut sum = f32x4::splat(0.0); // [0, 0, 0, 0]
    for i in (0..closest_under).step_by(4) {
        sum += f32x4::from_slice_unaligned(&x[i..]);
    }
    let mut final_sum = sum.sum();
    for i in closest_under..x.len() {
        final_sum += &x[i]
    }
    final_sum
}

/// parallel, from packed_simd with adjustment to handle non-4 sized
fn simd_sum_par(x: &[f32]) -> f32 {
    let closest_under = x.len() - (x.len() % 4);
    let mut final_sum: f32 = x[0..closest_under]
        .par_chunks(4)
        .map(f32x4::from_slice_unaligned)
        .sum::<f32x4>()
        .sum();
    for i in closest_under..x.len() {
        final_sum += &x[i]
    }
    final_sum
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use ndarray_npy::read_npy;

    use crate::ranking::simd_sandbox::{simd_sum, simd_sum_par};

    #[test]
    fn custom_reduce() {
        let q: Array1<f32> = read_npy("resources/query.npy").expect("require test file");
        assert_eq!(q.dim(), 293);
        assert_eq!(simd_sum(q.as_slice().unwrap()), q.sum());
        assert_eq!(
            simd_sum(q.as_slice().unwrap()),
            simd_sum_par(q.as_slice().unwrap())
        );
    }
}
