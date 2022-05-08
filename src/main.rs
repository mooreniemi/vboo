use colored::Colorize;
use ndarray::{Array, Array1, Array2};
use ranking::{
    op::Op,
    rank::{rank, rank_parallel},
    scorer::Scorer,
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
    /// Set scorer used to weight terms in document term matrix
    #[structopt(short, long, default_value = "bm25")]
    scorer: Scorer,
    /// Set source page
    #[structopt(short, long, default_value = "http://www.rust-lang.org/en-US/")]
    page: String,
    /// Set query string
    #[structopt(short, long, default_value = "rust company support")]
    query: String,
    /// Recreate test data using current query and page
    #[structopt(short, long)]
    fixture: bool,
    /// Exhaustively check op rather than use options
    #[structopt(short, long)]
    compare: bool,
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
    // in many cases (project gutenberg) it makes sense to remove these entirely
    let text = text.replace('\n', " ");

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

    let avg_sent_len: f32 = sent_lens.iter().sum::<usize>() as f32 / dx as f32;
    let k = 1.2;
    let b = 0.75;
    let start = Instant::now();
    // for vector boolean retrieval we need full sparse doc x term matrix
    // each row is a document, each column a term
    let mut doc_term_matrix = Array2::zeros((dx, tx));
    for (tidx, (_term, postings)) in inverted_idx.iter().enumerate() {
        for (postidx, freq) in postings.iter() {
            let tf = *freq as f32 / sent_lens[*postidx] as f32;
            let idf = (sents.len() as f32 / postings.len() as f32).ln();
            match opt.scorer {
                Scorer::TFIDF => {
                    doc_term_matrix[[*postidx, tidx]] = tf * idf;
                }
                Scorer::BM25 => {
                    let bm25 = idf
                        * (tf * (k + 1.0) / tf
                            + k * (1.0 - b + b * sent_lens[*postidx] as f32 / avg_sent_len));
                    // we hack in a scaling factor so we're beneath 1, otherwise AND breaks
                    doc_term_matrix[[*postidx, tidx]] = bm25 * 0.01;
                }
            }
        }
    }
    let duration = start.elapsed();
    println!(
        "Generating document {} x term {} matrix elapsed: {:?}",
        dx, tx, duration
    );
    if opt.fixture {
        ndarray_npy::write_npy(
            format!(
                "{}/{}",
                env!("CARGO_MANIFEST_DIR"),
                "resources/doc_term_matrix.npy"
            ),
            &doc_term_matrix,
        )
        .expect("wrote out dtm");
    }

    let start = Instant::now();
    // processing query like sents for term matching
    let q = opt.query.to_lowercase();
    let question = q
        .unicode_words()
        .map(|w| trim_clean(w))
        .collect::<HashSet<&str>>();
    let question_stemmed: HashSet<String> = question
        .iter()
        .map(|word| en_stemmer.stem(word.to_lowercase().as_str()).to_string())
        .collect();
    // embedding the query into the term space
    let mut query: Array1<f32> = Array::zeros(tx);
    for (tidx, (term, _postings)) in inverted_idx.iter().enumerate() {
        if question_stemmed.contains(&term.to_string()) {
            query[tidx] = 1.0;
        }
    }
    assert!(
        query.sum().gt(&0.0),
        "None of the query terms could be found in the document"
    );
    if question.len() as f32 != query.sum() {
        eprintln!(
            "Failed to find term in the document ({} != {})",
            question.len(),
            query.sum()
        );
    }
    if opt.fixture {
        ndarray_npy::write_npy(
            format!("{}/{}", env!("CARGO_MANIFEST_DIR"), "resources/query.npy"),
            &query,
        )
        .expect("wrote out query");
    }
    let duration = start.elapsed();
    println!("Embedding query elapsed: {:?}", duration);
    //dbg!(&query);

    // just testing out obvious outputs
    //dbg!(&sents[97], &doc_term_matrix.row(97));
    if opt.fixture {
        ndarray_npy::write_npy(
            format!("{}/{}", env!("CARGO_MANIFEST_DIR"), "resources/doc.npy"),
            &doc_term_matrix.row(97),
        )
        .expect("wrote out doc");
    }
    //let o = vboo::ranking::rank::or(&query.view(), &doc_term_matrix.row(97));
    //let a = vboo::ranking::rank::and(&query.view(), &doc_term_matrix.row(97));
    //dbg!(o, a);

    if opt.compare {
        println!("\nrank in parallel using both ops");

        // these are parallel within and so running them at same time won't be faster now
        let or_val = rank_parallel(&query.view(), &doc_term_matrix.view(), &Op::OR);
        let and_val = rank_parallel(&query.view(), &doc_term_matrix.view(), &Op::AND);

        let both = or_val.iter().zip(and_val.iter());

        for (idx, (result_or, result_and)) in both.enumerate() {
            println!("OR: {} - {:?}", &idx, &result_or);
            highlight(&sents[result_or.doc_id], &question_stemmed, &en_stemmer);
            println!("AND: {} - {:?}", &idx, &result_and);
            highlight(&sents[result_and.doc_id], &question_stemmed, &en_stemmer);
        }

        let correlation = kendalls::tau_b(
            &or_val.iter().map(|r| r.doc_id).collect::<Vec<usize>>(),
            &and_val.iter().map(|r| r.doc_id).collect::<Vec<usize>>(),
        )
        .unwrap();
        println!("rank correlation = {:?}", correlation);
    } else {
        // ranking results
        println!("\nrank in parallel using {:?}", opt.op);
        let topkv = rank_parallel(&query.view(), &doc_term_matrix.view(), &opt.op);
        for (idx, result) in topkv.iter().enumerate() {
            println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
        }
        println!(
            "\nrank in parallel, skimming for top results, using {:?}",
            opt.op
        );
        let topkv = rank_parallel_skim(&query.view(), &doc_term_matrix.view(), &opt.op);
        for (idx, result) in topkv.iter().enumerate() {
            println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
        }
        println!(
            "\nrank sequentially, skimming for top results, using {:?}",
            opt.op
        );
        let topk = rank(&query.view(), &doc_term_matrix.view(), &opt.op);
        let results = topk.into_sorted_vec();
        for (idx, result) in results.iter().enumerate() {
            println!("{} - {:?} - {}", &idx, &result, &sents[result.doc_id]);
            // dbg!(doc_term_matrix.row(result.doc_id));
        }
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

/// highlighting match terms
fn highlight(text: &str, terms: &HashSet<String>, en_stemmer: &Stemmer) {
    let colorized: Vec<String> = text
        .unicode_words()
        .map(|word| en_stemmer.stem(word.to_lowercase().as_str()).to_string())
        .map(|w| {
            if terms.contains(w.as_str()) {
                w.red().bold().to_string()
            } else {
                w.normal().to_string()
            }
        })
        .collect();
    println!("\"{}\"", colorized.join(" "))
}
