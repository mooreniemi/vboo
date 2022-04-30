use super::{op::Op, rank_result::RankResult};
use ndarray::{ArrayView1, ArrayView2};
use std::collections::BinaryHeap;

/// p for p-norm
static P: f64 = 2.0;

/// k of top k results
static K: usize = 10;

/// given an embedded query and a document x term matrix, rank by vboo op
pub fn rank(
    query: &ArrayView1<f64>,
    dt_matrix: &ArrayView2<f64>,
    op: Op,
) -> BinaryHeap<RankResult> {
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
