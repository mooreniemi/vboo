#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ndarray_npy::read_npy;
use vboo::ranking::rank::{and, or};

#[macro_use]
extern crate lazy_static;

lazy_static! {
    static ref DOC: Array1<f32> = read_npy(format!(
        "{}/{}",
        env!("CARGO_MANIFEST_DIR"),
        "resources/doc.npy"
    ))
    .expect("require test file");
    static ref Q: Array1<f32> = read_npy(format!(
        "{}/{}",
        env!("CARGO_MANIFEST_DIR"),
        "resources/query.npy"
    ))
    .expect("require test file");
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("or", |b| {
        b.iter(|| or(black_box(&Q.view()), black_box(&DOC.view())))
    });
    c.bench_function("and", |b| {
        b.iter(|| and(black_box(&Q.view()), black_box(&DOC.view())))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
