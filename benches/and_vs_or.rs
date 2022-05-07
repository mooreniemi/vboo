#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ndarray_npy::read_npy;
use vboo::ranking::rank::{and, or};

pub fn criterion_benchmark(c: &mut Criterion) {
    let doc: Array1<f64> = read_npy("resources/doc.npy").expect("require test file");
    let q: Array1<f64> = read_npy("resources/query.npy").expect("require test file");
    c.bench_function("or", |b| {
        b.iter(|| or(black_box(&q.view()), black_box(&doc.view())))
    });
    c.bench_function("and", |b| {
        b.iter(|| and(black_box(&q.view()), black_box(&doc.view())))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
