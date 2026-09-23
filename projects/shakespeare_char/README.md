# shakespeare_char

slmkit's **correctness reference** (DESIGN §5.0). It reproduces nanoGPT's character-level
Shakespeare run so that trainer bugs can be told apart from hard tasks: if slmkit cannot reach
roughly the published validation loss here, the problem is slmkit, not the data.

## Data

| | |
|---|---|
| Source | `tinyshakespeare/input.txt` from [karpathy/char-rnn](https://github.com/karpathy/char-rnn), the same file nanoGPT uses |
| Size | 1,115,394 characters, 65 distinct |
| Integrity | SHA-256 pinned in `project.py`; ingest fails if the upstream file changes |
| Licence | The text is Shakespeare's plays, which are **public domain**. The char-rnn repository declares no licence for the compiled file; slmkit does not redistribute it and downloads it at ingest time instead |

## Documents and splits

The file is one continuous text. `documents()` cuts it into contiguous blocks of ~10,000
characters, each ending at a blank line (a speech boundary). **Each block is its own `group`**,
so validation is ~10% of whole blocks (whole scenes), never lines scattered through training
text.

This differs from nanoGPT, which uses the last 10% of the file as validation. The two
validation sets are different text, so losses are comparable but not identical; the M1 pass
mark (≤ 1.55) allows for that.

`append_eos` is off: blocks are pieces of one text, not independent documents, so inserting an
end-of-document token between them would teach the model a boundary that does not exist.

## Graders

None. Validation loss is the grade, plus the generated samples printed at every eval.

## Known pitfalls

- The validation set contains a few characters that may not appear in training blocks. They
  encode as `<unk>`; `slm pack` reports the count as `val_unk_tokens`.
- The model will memorise: there is only ~1M characters of text and the run sees it ~80
  times. That is expected here; the loss curve is the point, not novelty.
