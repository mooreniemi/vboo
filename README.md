```
           ____
          /   /\
         /___/  \
        /   /\  /\
       /   /  \/  \
      /   /   /\   \
     /   /   /  \   \
    /   /   /\   \   \
   /   /   /  \   \   \
  /___/___/____\   \   \
 /   /          \   \  /\
/___/____________\___\/  \
\   \             \   \  /
 \___\_____________\___\/
```

# vboo

[Extended Boolean
model](https://en.wikipedia.org/wiki/Extended_Boolean_model) / vector
boolean retrieval in Rust. Purely for fun.

To run [iai](https://github.com/bheisler/iai) part of `cargo bench`, need
`valgrind` installed.

## stuff you can read on ebm

- http://www.minerazzi.com/tutorials/term-vector-6.pdf
- https://slideplayer.com/slide/16581216/


## cli

```
vboo 0.1.0
extended boolean model retrieval over a webpage

USAGE:
    vboo [FLAGS] [OPTIONS]

FLAGS:
    -c, --compare    Exhaustively check op rather than use options
    -f, --fixture    Recreate test data using current query and page
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -o, --op <op>            Set query op [default: or]
    -p, --page <page>        Set source page [default: http://www.rust-lang.org/en-US/]
    -q, --query <query>      Set query string [default: rust company support]
    -s, --scorer <scorer>    Set scorer used to weight terms in document term matrix [default: bm25]
```

### example usage

Basic:

```
cargo run -- --op and --query "the rust project receives support from companies through the donation of infrastructure"
```

Some use of `page` and `scorer`:

```
cargo run -- --op and --query "spanish square" --page "https://www.gutenberg.org/cache/epub/8442/pg8442.txt" --scorer tfidf
cargo run -- --op or --query "spanish square" --page "https://www.gutenberg.org/cache/epub/8442/pg8442.txt" --scorer tfidf
cargo run -- --op or --query "spanish square" --page "https://www.gutenberg.org/cache/epub/8442/pg8442.txt" --scorer bm25
cargo run -- --op and --query "spanish square" --page "https://www.gutenberg.org/cache/epub/8442/pg8442.txt" --scorer bm25
```

Using `compare`:

```
cargo run -- --query "large should there landlord" --page "https://www.gutenberg.org/files/1400/1400-0.txt" --scorer bm25 --compare
```
