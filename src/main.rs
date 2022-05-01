use ndarray::{Array, Array1, Array2};
use ranking::{
    op::Op,
    rank::{rank, rank_parallel},
};
use rust_stemmers::{Algorithm, Stemmer};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    time::Instant,
};
use structopt::StructOpt;
use unicode_segmentation::UnicodeSegmentation;
use webpage::{Webpage, WebpageOptions};

use crate::ranking::rank::rank_parallel_skim;

mod ranking;

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
    let start = Instant::now();
    let wpage = Webpage::from_url(opt.page.as_str(), WebpageOptions::default())
        .expect("Could not read from URL");
    let duration = start.elapsed();
    println!("Downloading page elapsed: {:?}", duration);
    let text = wpage.html.text_content;

    let start = Instant::now();
    // process text into sentences
    let sents = text
        .unicode_sentences()
        .map(|s| trim_clean(s))
        .collect::<Vec<&str>>();
    let duration = start.elapsed();
    println!("Sentence splitting elapsed: {:?}", duration);

    let mut sent_lens = Vec::new();
    let dx = sents.len();
    let en_stemmer = Stemmer::create(Algorithm::English);

    let start = Instant::now();
    // it's convenient especially for debugging to have a normal inverted index
    //----------------------------------------(doc  , freq )
    let mut inverted_idx: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (idx, sent) in sents.iter().enumerate() {
        let words = sent.unicode_words().collect::<Vec<&str>>();
        //-----------------------------term  , freq
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
    let duration = start.elapsed();
    println!("Inverting the index elapsed: {:?}", duration);
    // dbg!(&inverted_idx);
    let tx = inverted_idx.len();

    let start = Instant::now();
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
    let duration = start.elapsed();
    println!("Generating document x term matrix elapsed: {:?}", duration);
    // for writing out test data
    // ndarray_npy::write_npy("/tmp/doc_term_matrix.npy", &doc_term_matrix).expect("wrote out dtm");

    let start = Instant::now();
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
    // for writing out test data
    // ndarray_npy::write_npy("/tmp/query.npy", &query).expect("wrote out q");
    let duration = start.elapsed();
    println!("Embedding query elapsed: {:?}", duration);

    // just testing out obvious outputs
    //dbg!(&sents[97], &doc_term_matrix.row(97));
    //let o = or(&query.view(), &doc_term_matrix.row(97));
    //let a = and(&query.view(), &doc_term_matrix.row(97));
    //dbg!(o, a);

    // ranking results
    let topkv = rank_parallel(&query.view(), &doc_term_matrix.view(), &opt.op);
    for (idx, result) in topkv.iter().enumerate() {
        println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
    }
    let topkv = rank_parallel_skim(&query.view(), &doc_term_matrix.view(), &opt.op);
    for (idx, result) in topkv.iter().enumerate() {
        println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
    }
    let topk = rank(&query.view(), &doc_term_matrix.view(), &opt.op);
    let results = topk.into_sorted_vec();
    for (idx, result) in results.iter().enumerate() {
        println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
    }

    Ok(())
}

/// removing whitespace and newlines on both ends
fn trim_clean(input: &str) -> &str {
    input
        .strip_suffix("\r\n")
        .or(input.strip_suffix("\n"))
        .unwrap_or(input)
        .trim()
}
