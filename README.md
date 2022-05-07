# vboo

[Extended Boolean
model](https://en.wikipedia.org/wiki/Extended_Boolean_model) / vector
boolean retrieval in Rust. Purely for fun.

To run [iai](https://github.com/bheisler/iai) part of `cargo bench`, need
`valgrind` installed.

# cli

```
vboo 0.1.0
extended boolean model retrieval over a webpage

USAGE:
    vboo [OPTIONS]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -o, --op <op>          Set query op [default: or]
    -p, --page <page>      Set source page [default: http://www.rust-lang.org/en-US/]
    -q, --query <query>    Set query string [default: rust company support]
```

## example usage

```
cargo run -- --op and --query "the rust project receives support from companies through the donation of infrastructure"
```
