use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2};
use rust_stemmers::{Algorithm, Stemmer};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    str::FromStr,
};
use structopt::StructOpt;
use unicode_segmentation::UnicodeSegmentation;
use webpage::{Webpage, WebpageOptions};

/// p for p-norm
static P: f64 = 2.0;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "vboo",
    about = "extended boolean model retrieval over a webpage"
)]
struct Opt {
    /// Set query op
    #[structopt(short, long, default_value = "or")]
    op: Op,
    /// Set source page
    #[structopt(short, long, default_value = "http://www.rust-lang.org/en-US/")]
    page: String,
    /// Set query string
    #[structopt(short, long, default_value = "rust company support")]
    query: String,
}

fn main() -> Result<(), &'static str> {
    let opt = Opt::from_args();

    // get some content to process
    let wpage = Webpage::from_url(opt.page.as_str(), WebpageOptions::default())
        .expect("Could not read from URL");
    let text = wpage.html.text_content;

    // process text into sentences
    let sents = text
        .unicode_sentences()
        .map(|s| trim_clean(s))
        .collect::<Vec<&str>>();

    let mut sent_lens = Vec::new();
    let dx = sents.len();
    let en_stemmer = Stemmer::create(Algorithm::English);

    // it's convenient especially for debugging to have a normal inverted index
    //----------------------------------------(doc  , freq )
    let mut inverted_idx: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (idx, sent) in sents.iter().enumerate() {
        let words = sent.unicode_words().collect::<Vec<&str>>();
        //-------------------------term  , freq
        let mut stemmed_words: HashMap<String, usize> = HashMap::new();
        for word in words.iter() {
            let sword = en_stemmer.stem(word.to_lowercase().as_str()).to_string();
            *stemmed_words.entry(sword.clone()).or_insert(0) += 1;
        }
        sent_lens.push(stemmed_words.len());
        for (sword, freq) in stemmed_words.iter() {
            inverted_idx
                .entry(sword.clone())
                .or_insert(Vec::new())
                //----(docid,freq)
                .push((idx, *freq));
        }
    }
    // dbg!(&inverted_idx);
    let tx = inverted_idx.len();

    // for vector boolean retrieval we need full sparse doc x term matrix
    // each row is a document, each column a term
    let mut doc_term_matrix = Array2::zeros((dx, tx));
    for (tidx, (_term, postings)) in inverted_idx.iter().enumerate() {
        for (postidx, freq) in postings.iter() {
            let tf = *freq as f64 / sent_lens[*postidx] as f64;
            let idf = (sents.len() as f64 / postings.len() as f64).ln();
            doc_term_matrix[[*postidx, tidx]] = tf * idf;
        }
    }

    // processing query like sents for term matching
    let q = opt.query.to_lowercase();
    let question = q
        .unicode_words()
        .map(|w| trim_clean(w))
        .collect::<HashSet<&str>>();
    // embedding the query into the term space
    let mut query: Array1<f64> = Array::zeros(tx);
    for (tidx, (term, _postings)) in inverted_idx.iter().enumerate() {
        if question.contains(&term.as_str()) {
            query[tidx] = 1.0;
        }
    }

    // just testing out obvious outputs
    //dbg!(&sents[97], &doc_term_matrix.row(97));
    //let o = or(&query.view(), &doc_term_matrix.row(97));
    //let a = and(&query.view(), &doc_term_matrix.row(97));
    //dbg!(o, a);

    // ranking results
    let mut topk = rank(&query.view(), &doc_term_matrix.view(), opt.op);
    let mut results = Vec::new();
    while let Some(result) = topk.pop() {
        results.push(result);
    }
    // results are in minheap order so we flip them around
    results.reverse();
    for (idx, result) in results.iter().enumerate() {
        println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
    }

    Ok(())
}

#[derive(Debug)]
struct RankResult {
    doc_id: usize,
    score: f64,
}

impl PartialEq for RankResult {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for RankResult {}

impl PartialOrd for RankResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // reverse order for min heap behavior during ranking
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for RankResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn rank(query: &ArrayView1<f64>, dt_matrix: &ArrayView2<f64>, op: Op) -> BinaryHeap<RankResult> {
    // dbg!(query.dim(), dt_matrix.dim());
    // this just preallocates memory for the heap, doesn't enforce max len
    let mut topk: BinaryHeap<RankResult> = BinaryHeap::with_capacity(10);
    for (doc_id, doc) in dt_matrix.rows().into_iter().enumerate() {
        let score = match op {
            Op::AND => and(&query, &doc),
            Op::OR => or(&query, &doc),
        };
        if let Some(min) = topk.peek() {
            if score.gt(&min.score) {
                if topk.len().eq(&10) {
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

#[derive(Debug, PartialEq)]
enum Op {
    AND,
    OR,
}

impl FromStr for Op {
    type Err = String;

    fn from_str(input: &str) -> Result<Op, Self::Err> {
        match input {
            "or" => Ok(Op::OR),
            "and" => Ok(Op::AND),
            _ => Err("unsupported ranking operation".to_string()),
        }
    }
}

/// sqrt((w1^2 + w2^2)/p=2)
fn or(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert!(a.dim().eq(&b.dim()));
    let c = a * b;
    let c = c.map(|e| e.powf(P));
    (c.sum() / a.dim() as f64).sqrt()
}

/// 1 - sqrt(((1-w1)^2 + (1-w2)^2)/p=2)
fn and(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    assert!(a.dim().eq(&b.dim()));
    let c = a * b;
    let d = c.map(|e| (1.0 - e).powf(P));
    1.0 - (d.sum() / a.dim() as f64).sqrt()
}

/// removing whitespace and newlines on both ends
fn trim_clean(input: &str) -> &str {
    input
        .strip_suffix("\r\n")
        .or(input.strip_suffix("\n"))
        .unwrap_or(input)
        .trim()
}
