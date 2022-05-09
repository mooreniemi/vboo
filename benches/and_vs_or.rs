#![allow(unused)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ndarray_npy::read_npy;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use vboo::ranking::rank::{and, or};

pub fn criterion_benchmark(c: &mut Criterion) {
    let q = Array1::random(256, Uniform::<f32>::new(0., 1.));
    let d = Array1::random(256, Uniform::<f32>::new(0., 1.));
    // println!("{:8.4}", d);

    c.bench_function("or", |b| {
        b.iter(|| or(black_box(&q.view()), black_box(&d.view())))
    });
    c.bench_function("and", |b| {
        b.iter(|| and(black_box(&q.view()), black_box(&d.view())))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
