use super::{op::Op, rank_result::RankResult};
use ndarray::parallel::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::{collections::BinaryHeap, time::Instant};

/// p for p-norm
static P: f64 = 2.0;

/// k of top k results
static K: usize = 10;

/// given an embedded query and a document x term matrix, rank by vboo op
pub fn rank_parallel_skim(
    query: &ArrayView1<f64>,
    dt_matrix: &ArrayView2<f64>,
    op: &Op,
) -> Vec<RankResult> {
    let start = Instant::now();
    // dbg!(query.dim(), dt_matrix.dim());
    let pi = dt_matrix
        .rows()
        .into_iter()
        .enumerate()
        .par_bridge()
        .into_par_iter();

    // can't directly collect into BinaryHeap
    let results: Vec<BinaryHeap<RankResult>> = pi
        .fold(
            || BinaryHeap::new(),
            |mut topk: BinaryHeap<RankResult>, (doc_id, doc)| {
                let score = match op {
                    Op::AND => and(&query, &doc),
                    Op::OR => or(&query, &doc),
                };
                let rr = RankResult { doc_id, score };
                if let Some(min) = topk.peek() {
                    if score.gt(&min.score) {
                        if topk.len().eq(&K) {
                            topk.pop();
                        }
                        let rr = RankResult { doc_id, score };
                        topk.push(rr);
                    }
                } else {
                    topk.push(rr);
                }
                topk
            },
        )
        .collect();

    // we now have to rejoin the split up results
    let mut returned = BinaryHeap::new();
    for bh in results.iter() {
        for e in bh {
            returned.push(e.clone());
        }
    }

    let returned = returned
        .into_sorted_vec()
        .iter()
        .take(K)
        .map(|e| *e)
        .collect();
    let duration = start.elapsed();
    println!("Time elapsed in rank_parallel_skim() is: {:?}", duration);

    returned
}

/// given an embedded query and a document x term matrix, rank by vboo op
pub fn rank_parallel(
    query: &ArrayView1<f64>,
    dt_matrix: &ArrayView2<f64>,
    op: &Op,
) -> Vec<RankResult> {
    let start = Instant::now();
    // dbg!(query.dim(), dt_matrix.dim());
    let pi = dt_matrix
        .rows()
        .into_iter()
        .enumerate()
        .par_bridge()
        .into_par_iter();

    let mut results: Vec<RankResult> = pi
        .map(|(doc_id, doc)| {
            let score = match op {
                Op::AND => and(&query, &doc),
                Op::OR => or(&query, &doc),
            };
            let rr = RankResult { doc_id, score };
            rr
        })
        .collect();

    // this sorts everything rather than having the topk heap
    // but there'd be overhead in reusing the same topk heap across threads
    results.par_sort();
    let duration = start.elapsed();
    println!("Time elapsed in rank_parallel() is: {:?}", duration);

    results.iter().take(K).map(|e| *e).collect()
}

/// given an embedded query and a document x term matrix, rank by vboo op
pub fn rank(
    query: &ArrayView1<f64>,
    dt_matrix: &ArrayView2<f64>,
    op: &Op,
) -> BinaryHeap<RankResult> {
    let start = Instant::now();
    // dbg!(query.dim(), dt_matrix.dim());
    // this just preallocates memory for the heap, doesn't enforce max len
    let mut topk: BinaryHeap<RankResult> = BinaryHeap::with_capacity(K);
    for (doc_id, doc) in dt_matrix.rows().into_iter().enumerate() {
        let score = match op {
            Op::AND => and(&query, &doc),
            Op::OR => or(&query, &doc),
        };
        if let Some(min) = topk.peek() {
            if score.gt(&min.score) {
                if topk.len().eq(&K) {
                    topk.pop();
                }
                let rr = RankResult { doc_id, score };
                topk.push(rr);
            }
        } else {
            let rr = RankResult { doc_id, score };
            topk.push(rr);
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed in rank() is: {:?}", duration);
    topk
}

/// sqrt((w1^2 + w2^2)/p=2)
pub fn or(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert!(a.dim().eq(&b.dim()));
    let c = a * b;
    let c = c.map(|e| e.powf(P));
    (c.sum() / a.dim() as f64).sqrt()
}

/// 1 - sqrt(((1-w1)^2 + (1-w2)^2)/p=2)
pub fn and(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert!(a.dim().eq(&b.dim()));
    let c = a * b;
    let d = c.map(|e| (1.0 - e).powf(P));
    1.0 - (d.sum() / a.dim() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use ndarray_npy::read_npy;

    use crate::ranking::{
        op::Op,
        rank::{rank, rank_parallel, rank_parallel_skim},
    };

    #[test]
    fn all_rank_same_top_one() {
        let dtm: Array2<f64> =
            read_npy("resources/doc_term_matrix.npy").expect("require test file");
        assert_eq!(dtm.dim(), (126, 293));
        let q: Array1<f64> = read_npy("resources/query.npy").expect("require test file");
        assert_eq!(q.dim(), 293);
        let rp = rank_parallel(&q.view(), &dtm.view(), &Op::AND);
        let rps = rank_parallel_skim(&q.view(), &dtm.view(), &Op::AND);
        let r = rank(&q.view(), &dtm.view(), &Op::AND).into_sorted_vec();
        assert_eq!(rp.get(0), rps.get(0));
        assert_eq!(rp.get(0), r.get(0));
    }
}
