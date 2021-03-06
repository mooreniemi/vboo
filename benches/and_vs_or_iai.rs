#![allow(unused)]
use iai::{black_box, main};
use ndarray::Array1;
use ndarray_npy::read_npy;
use vboo::ranking::rank::{and, or};

#[macro_use]
extern crate lazy_static;

lazy_static! {
    // iai only runs the functions once and is not able to exclude setup code
    // https://github.com/bheisler/iai/issues/23, https://github.com/bheisler/iai/issues/20
    // so rather than generate a random vector, just use stored fixtures
    // in general iai looks to be inactive, so this is just for fun really
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
