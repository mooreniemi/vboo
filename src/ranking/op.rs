use std::str::FromStr;

#[derive(Debug, PartialEq)]
pub enum Op {
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
