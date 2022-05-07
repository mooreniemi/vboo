use std::cmp::Ordering;

#[derive(Debug, Copy, Clone)]
pub struct RankResult {
    pub doc_id: usize,
    pub score: f32,
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
