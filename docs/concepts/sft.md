# Supervised fine-tuning: from continuing text to following requests

*Milestone M2, Phase D. Code: [`src/slmkit/data/sft.py`](../../src/slmkit/data/sft.py) (masking),
[`src/slmkit/train/stage.py`](../../src/slmkit/train/stage.py) (the SFT stage),
[`projects/abc_music/project.py`](../../projects/abc_music/project.py) (the requests). Terms:
[`../GLOSSARY.md`](../GLOSSARY.md). Hands-on: [`../runbooks/m2-abc-music.md`](../runbooks/m2-abc-music.md) §D.*

A pretrained model **continues text**. Give it `R:jig / M:6/8 / K:G` and it writes a jig, because
tunes that start like that continue like that. It doesn't **follow requests**: asked in words, "a jig
in G major", it has never seen English, so it guesses. Supervised fine-tuning (SFT) teaches the second
behaviour, with a short extra round of training on request → answer pairs.

This is the same step that turns a base language model into a chat assistant, at a tiny scale: a base
model trained on the internet continues documents, and SFT on conversations teaches it to answer. (Chat
assistants usually add a further stage, reinforcement learning from human feedback; slmkit doesn't.)

---

## 1. What an SFT example looks like

Each tune becomes one request and one answer. The request is built from the tune's own headers, in one
of a few phrasings chosen per tune by hash:

```
% compose a reel in the key of Bb major        ← the prompt (request)
R:reel                                         ← the answer: every header, then the tune
M:2/4
L:1/16
K:Bb
vB2(Bd) fBdf | bdfb Td'2(c'b) | gbfb dfbg | …
<eos>
```

- **The request is an ABC comment** (`% …`). Request plus answer is still a valid ABC file, so the
  graders and `abc2midi` can read it unchanged.
- **The answer always carries every header**, even if header dropout removed some from that tune's
  pretraining copy. The model should *state* the rhythm, meter and key it was asked for, which makes its
  interpretation of the request visible.
- **Transposed copies are included** (train split only), so requests cover more keys: 11,312 training
  examples from 3,829 tunes.

### The vocabulary is frozen after pretraining

The char tokenizer was built from ABC text, which contains no `q`, no capital `J`, `N`, `W`, `X` or `Y`,
and no `?`. A request like "Write a jig?" would turn the `W` and the `?` into `<unk>`, a token the model
never learned anything about. So every request template starts lower-case and avoids those characters,
and a test checks every template against every rhythm and key in the corpus.

This is a real constraint, not a quirk: **you can't add vocabulary after pretraining without also
adding untrained embedding rows**. Production tokenizers avoid the problem by including every possible
byte from the start. **Rejected:** adding special tokens such as `<request>` for SFT. They would need
new, untrained embedding rows, and a model with 0.86M parameters would learn them from scratch during
a 30-second fine-tune.

---

## 2. The one essential difference: loss only on the answer

Pretraining scores every next-token prediction. SFT scores only the **answer**:

```
tokens    %  a     j  i  g  …  \n R  :  j  i  g  \n  …  <eos>
target    a  ' '   i  g  …  \n R  :  j  i  g  \n M  …
loss      ✗  ✗     ✗  ✗  ✗  ✗  ✓  ✓  ✓  ✓  ✓  ✓  ✓  …
```

The targets under ✗ are set to `-100`, which PyTorch's cross-entropy ignores (`IGNORE_INDEX`, already
tested in M1). The first answer token (`R`) is predicted *from* the last request token, so it counts.
Padding at the end of each row is ignored the same way.

**Why mask:** the request is short, formulaic, and never generated at inference time. Training on it
spends learning signal on a skill nobody will use, and teaches the model to *write* requests. With the
mask, all of the gradient goes into answering. `test_prompt_positions_get_zero_loss_and_zero_gradient`
checks both: masked positions contribute exactly zero loss, and exactly zero gradient.

**Rows, not a stream.** Pretraining reads random windows from one long concatenation. SFT puts exactly
one example per row, padded on the right to the context length. Because attention is causal, real
tokens never see the padding after them. 407 of 11,312 examples (3.6%) are longer than the 512-token
context and are skipped rather than cut off mid-tune.

---

## 3. Settings: start from the base, change it gently

| Setting | Pretraining | SFT | Why |
|---|---|---|---|
| starting weights | random | the pretrained run's `ckpt/best` | SFT refines a model that already writes tunes |
| optimizer state | — | fresh | AdamW's running averages describe pretraining gradients, not this task's |
| learning rate | 1e-3 | **3e-4** | adjust the model, don't overwrite what pretraining learned |
| warmup | 1M tokens | 100K tokens | the model is already trained; only the new optimizer needs a gentle start |
| batch | 64 | 32 | fewer, more varied updates per epoch |
| budget | 30M tokens | **3 epochs** ≈ 16.8M tokens | a pass count, converted to tokens like everything else |

Settings come from `presets/train/sft.yaml`, then the experiment's `sft:` section. The SFT run gets its
own directory and ID (config + `sft:` + the parent run's ID), resumes exactly like pretraining, and
`slm run` continues into it when `sft.enabled` is true.

Both stages use **the same trainer**; only the batches, starting weights and settings differ
(`train/stage.py`). **Rejected:** a separate SFT training loop. It would duplicate resume, checkpoints,
guards and logging, and duplicated infrastructure drifts apart.

---

## 4. Did it work? Parity is the pass mark

Measured on `abc_music/baseline` (char, `nano`), 3 sampling seeds × 200 samples each, graded by the
same graders against what was **asked**:

| | Base model, header prompt | Base model, asked in words | **SFT model, asked in words** |
|---|---|---|---|
| plays | 0.643 ± 0.025 | 0.413 ± 0.051 | **0.635** ± 0.059 |
| bar_accuracy | 0.677 ± 0.011 | 0.268 ± 0.025 | **0.689** ± 0.011 |
| ends_on_tonic | 0.188 ± 0.058 | 0.132 ± 0.010 | **0.292** ± 0.046 |
| ended (finished itself) | 0.920 ± 0.005 | 0.830 ± 0.013 | **0.982** ± 0.010 |
| novelty | 0.999 | 0.991 | 0.998 |

Reading it:

- **The middle column is why SFT exists.** Asked in words, the base model's bar accuracy falls from
  0.68 to 0.27: it doesn't understand the request.
- **SFT reaches parity** with the base model's header prompt on playing and bars. That is the pass mark
  DESIGN §5.1 set, and it is what was expected. Header dropout already taught the base model to follow
  headers, so SFT's contribution is the **interface**, not better music.
- **Two side effects are real improvements.** SFT examples are always complete tunes, so the fine-tuned
  model finishes 98% of the time, and ends on the tonic more often (0.29 vs 0.19, a gap of about twice
  the spread). One training seed, though; Phase F repeats it.

What it looks like. Both samples were asked, in words, for "a jig in G major":

```
Before SFT (the pretrained model)            After SFT
% a jig in G major                           % a jig in G major
M:6/8                                        R:jig
L:1/8                                        M:6/8
K:D                ← wrong key               L:1/8
F EFG A2 B | cde d2 B AFC | …                K:G
                                             D|DFD DFG | ABA ABA | G2 E GED | AFA ABA |
                                             DED GEC | DBB ABd | cBA GEG | AFA A2 :|
```

---

## 5. What SFT can't do here

- **Teach new music.** Everything the fine-tuned model writes, it learned in pretraining; SFT redirects
  it. That's why graders show parity rather than a jump.
- **Understand arbitrary English.** The model has seen eight phrasings built from a few words. "Give me
  something cheerful" means nothing to it. The requests are a small, closed language, and only
  scale changes that.
- **Fix the tonic problem.** Ending on the tonic improved, but 0.29 is still far below real tunes' 0.80.
  That's a model-size limit (evaluation.md §5), which Phase F measures.
