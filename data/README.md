# Brown corpus: prepared learning dataset

This directory contains a **locally prepared version of the Brown corpus** for
word-level next-token prediction. It uses the same corpus as one of the Bengio
et al. (2003) experiments, but **not the authors' exact preprocessing, document
order, vocabulary, or released splits**. Do not compare its perplexity directly
with the paper's tables.

## Files

| Path | Contents | Count |
| --- | --- | ---: |
| `training/brown.txt` | Training token stream | 800,000 tokens |
| `eval/brown_validation.txt` | Validation token stream | 200,000 tokens |
| `eval/brown_test.txt` | Held-out test token stream | 161,192 tokens |
| `vocab.txt` | Ordered vocabulary | 17,000 entries |
| `manifest.json` | Source, preprocessing, counts, and checksums | |
| `raw/brown.zip` | Original NLTK Brown archive | 500 documents |
| `raw/BROWN_README.txt` | Source attribution and redistribution notice | |
| `raw/BROWN_CONTENTS.txt` | Source document descriptions and credits | |

`model/` is reserved for your model artifacts; no model files were created.

## Format and token IDs

All prepared text files are UTF-8 with **one token per line**. They are already
tokenized. A line is a token, not a sentence. Case and punctuation are preserved.
There is no header and no blank separator line.

In `vocab.txt`, the zero-based line position is the token ID:

- First line: `<unk>`; ID `0`.
- The remaining 16,999 entries are corpus tokens, including punctuation.
- Ordering: descending frequency in training data; ties are broken lexically.

Use this **same mapping** for training, validation, test, and inference, and
preserve it alongside saved model weights. Set the model's vocabulary size to
17,000. The vocabulary contains token strings, not pretrained embeddings;
your model learns those vectors.

All tokens outside the selected vocabulary have already been replaced with
`<unk>` in the three prepared streams. Every token in these files therefore has
an entry in `vocab.txt`.

## How this version was prepared

1. Download the NLTK Brown archive linked below.
2. Read its 500 document paths in lexicographic order: `ca01` through `cr09`.
3. Split each document on whitespace. Each original item is `word/POS-tag`;
   remove the final slash and POS-tag suffix, preserving slashes within words.
4. Preserve capitalization, punctuation, and original token order. Do not add
   sentence, paragraph, document, or padding tokens.
5. Concatenate the documents into a 1,161,192-token stream, without shuffling.
6. Take the first 800,000 tokens for training, the next 200,000 for validation,
   and the remainder for testing. Boundaries can fall inside a document or
   sentence. The sorted genre order means the splits have different genre mixes;
   this is not a random or stratified split.
7. Build the vocabulary from **training tokens only**: reserve `<unk>` at ID 0
   and select the 16,999 most frequent training tokens, with lexical tie breaks.
8. Map excluded tokens to `<unk>` independently in each split, then write one
   token per line.

Do not build training examples by joining training and evaluation files.
Do not fit a new vocabulary on evaluation data. Use validation for model
selection; reserve test for the final evaluation.

Unknown-token rates: training 4.68%, validation 8.42%, test approximately 8.39%.

## Differences from the reference paper

The paper's Brown experiment reports 1,181,041 tokens and a vocabulary of
**16,383**, obtained by merging words with frequency at most three into a single
symbol. Its test split contains 181,041 tokens. The paper also mentions text and
paragraph separation marks that this preparation does not add.

The NLTK source used here has 1,161,192 tokens before unknown-word replacement,
so retaining the paper's training and validation token counts leaves 161,192 test
tokens. This preparation deliberately uses **17,000 vocabulary entries** to match
the requested learning setup, rather than claiming the paper used exactly 17,000.

The paper's separate AP News experiment has 17,964 vocabulary entries. That
corpus and its original processed files were not obtained here. Karpathy's
*makemore* names dataset is also not included; it is a different, character-level
experiment.

## Sources and redistribution

- Paper, especially Section 4:
  https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- NLTK Brown corpus archive:
  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
- NLTK corpus collection:
  https://github.com/nltk/nltk_data

The source README credits W. N. Francis and H. Kucera, Brown University, and states:
"Distributed with the permission of the copyright holder, redistribution
permitted." The original notice and document-level credits are retained under
`raw/`. Source and prepared-file SHA-256 checksums are recorded in `manifest.json`.
