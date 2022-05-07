#![allow(unused)]
use iai::{black_box, main};
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

fn iai_benchmark_and() -> f32 {
    and(black_box(&Q.view()), black_box(&DOC.view()))
}

fn iai_benchmark_or() -> f32 {
    or(black_box(&Q.view()), black_box(&DOC.view()))
}

iai::main!(iai_benchmark_and, iai_benchmark_or);
