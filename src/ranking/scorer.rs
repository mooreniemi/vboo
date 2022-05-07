use std::str::FromStr;

#[derive(Debug, PartialEq)]
pub enum Scorer {
    /// More sensitive to document length because no normalization
    TFIDF,
    /// Less sensitive to document length due to normalization
    BM25,
}

impl FromStr for Scorer {
    type Err = String;

    fn from_str(input: &str) -> Result<Scorer, Self::Err> {
        match input {
            "tfidf" => Ok(Scorer::TFIDF),
            "tf_idf" => Ok(Scorer::TFIDF),
            "tf-idf" => Ok(Scorer::TFIDF),
            "bm25" => Ok(Scorer::BM25),
            _ => Err("unsupported scorer".to_string()),
        }
    }
}
