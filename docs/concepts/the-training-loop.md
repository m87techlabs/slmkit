# The training loop

*Milestone M1, Phase C. Code: [`src/slmkit/train/`](../../src/slmkit/train/). Terms:
[`../GLOSSARY.md`](../GLOSSARY.md). Check it on your machine:
[`../runbooks/m1-engine.md`](../runbooks/m1-engine.md) §C.*

Phase A turned text into token IDs; Phase B built a model that turns token IDs into predictions.
This page covers the loop that makes the predictions good, and everything around it that lets a
run survive a machine that is switched off every night.

All numbers are from the real `shakespeare_char/ref` run on the reference machine.

---

## 1. One step, in five lines

```python
lr = lr_at(tokens_seen, ...)                   # 1. how big a step to take right now (§3)
_, loss = model(x, y)                          # 2. forward: how wrong are the predictions?
loss.backward()                                # 3. backward: which way should each weight move?
clip_grad_norm_(model.parameters(), 1.0)       # 4. cap the step if the gradient is unusually large
optimizer.step()                               # 5. move every weight a little (§4)
```

- **Forward** runs the batch through the model and computes the loss: the average of
  −ln(probability of the correct next token) over 64 × 256 = 16,384 positions.
- **Backward** (backpropagation) computes, for each of the 10.6M weights, how the loss would change
  if that weight changed slightly. That set of numbers is the **gradient**. PyTorch derives it
  automatically from the forward computation, which is what "autograd" means.
- **Clipping**: the gradient's overall size (its norm) is usually around 0.3–0.5 in this run. If one
  unlucky batch produces 50, the step would be 100× larger than usual and could undo hours of
  progress. Clipping rescales any gradient with norm above 1.0 down to exactly 1.0, keeping its
  direction.
- **Guard**: before stepping, the loss and gradient norm are checked for NaN or infinity. A single
  NaN written into the weights is permanent, because every later forward pass produces NaN. So a bad
  step is skipped and logged, and 20 bad steps in a row abort the run with advice
  (`train/guards.py`).

Everything else in `trainer.py` is bookkeeping around those five lines.

---

## 2. What "training" looks like from outside

The first minute of the reference run, one line every 100 steps (1.6M tokens):

```
eval  step      0  train 4.3709  val 4.3689         ← ln 67 = 4.20: guessing uniformly
── sample @ step 0: ROMEO
ROMEO:
Azz;BdL:klq&vJgYbkmB�Q:oHICrqqBo                    ← random characters

eval  step    100  train 1.8863  val 1.8996         ← already past "knows letter pairs" (2.45)
── sample @ step 100: ROMEO
ROMEO:
As so now to here. Buptore, the eforthit.

GLOUCESTER:
I thou word is butiencely broous peaciour have or trom,

eval  step    300  train 1.3998  val 1.4568         ← nanoGPT's published best is ~1.47
── sample @ step 300: ROMEO
ROMEO:
All you know him know you, this enjoy!

Nurse:
Ay, sir, the captition, you be lown the Duke of;
```

Within 100 steps it has the *format* (speaker names in capitals, a colon, a line break, verse
line lengths). By 300 it spells most words and uses real character names. What it never gets,
at this size, is *meaning*.

The `�` at step 0 is the `<unk>` token. The untrained model assigns it the same probability as
everything else; since `<unk>` never occurs in training text, the model learns within a few steps
never to produce it.

### When results look too good, suspect a bug first

Validation loss 1.457 after 300 steps (8% of the run) matches nanoGPT's *final* number. CONTRIBUTING.md
says to suspect leakage before celebrating, so we checked:

| Check | Result | What it rules out |
|---|---|---|
| Change tokens 128+, compare logits at 0–127, on the **compiled bf16 GPU path** | max change 0.0 | The causal mask leaking under compile or autocast (the CPU test didn't cover that path) |
| Look for validation text in training text: 50-character windows, every 10th position | 0 of 11,148 found | Duplicated passages across the split |
| Read the samples | Quality matches the loss | Loss computed wrongly: leakage makes loss look good while samples stay bad |

All clean. The honest explanation is that the run sees its 1M-character corpus ~80 times, and a
Llama-style model converges quickly. The real risk is what happens *next*: training loss keeps
falling while validation loss rises, i.e. **overfitting**. That is why the trainer keeps a separate
`best` checkpoint, chosen by validation loss, and not just the last one.

### What the full run showed

The complete reference run (Phase D in the runbook) bottomed out at **val 1.2888 at step 1250**,
about 20 passes over the training text, then overfit: by step 5000 training loss was 0.44 and
validation 1.63. Two lessons, both measured:

- **The final checkpoint was the worst one.** Validation loss rose from step 1250 to the end. That is
  why `ckpt/best` is kept separately and is what `slm sample` loads by default.
- **Rising validation loss did not mean copying.** Samples from the final checkpoint contain no
  40-character passage from the training text, just like the best one. The model became
  over-confident about training-specific patterns; its text still reads as plausible verse. Loss and
  samples measure different things, and "it's memorizing" had to be checked, not assumed.

---

## 3. The learning-rate schedule: warmup, then cosine, in tokens

The learning rate (lr) is the size of each step. `train/schedule.py` makes it a pure function of
tokens seen:

```
lr
1e-3 │   ╭───╮
     │  ╱     ╲____
     │ ╱           ╲______
1e-4 │╱                   ╲__________   ← floor: 10% of peak
     └┴──────────────────────────────── tokens
      1.6M                          81.9M
```

With the `ref` settings (peak 1e-3, warmup 1,638,400 tokens = 100 steps, floor 1e-4), printed by
the schedule code itself. Each step uses the lr for the tokens it will have seen *after* that step:

| Step | Tokens after the step | lr | Phase |
|---|---|---|---|
| 0 | 16,384 | 1.00e-05 | first step: 1/100 of peak |
| 49 | 819,200 | 5.00e-04 | halfway up |
| 99 | 1,638,400 | 1.00e-03 | peak |
| 2,549 | 41,779,200 | 5.50e-04 | halfway down the cosine: midpoint of peak and floor |
| 4,999 | 81,920,000 | 1.00e-04 | floor |

**A bug this table caught.** The first version computed warmup as `lr × (tokens + 1) / warmup`,
copying nanoGPT's `(step + 1) / warmup_steps`. In steps the `+1` makes step 0 non-zero; in tokens it
adds one token out of 1.6 million, so step 0 ran at **6×10⁻¹⁰**, effectively a wasted step. Writing
this table from the code, instead of from memory, is what exposed it.

- **Why warmup:** at step 0 the weights are random and AdamW has no history of gradient sizes, so its
  first updates are poorly scaled. A full-size lr then can push the model somewhere it never
  recovers from. 100 gentle steps avoid that for a negligible cost.
- **Why decay:** big steps make fast early progress but keep bouncing around a good solution; small
  late steps let it settle. Cosine is the smooth, standard way to get from one to the other.
- **Why tokens rather than steps:** double the batch size and a run has half as many steps but sees
  the same data. A schedule in tokens still warms up over the same 1.6M tokens either way. Tokens
  are the unit that matters, which is why every budget in slmkit is in tokens too.

Because it is a pure function, the schedule needs no saved state: restoring `tokens_seen` from a
checkpoint restores the lr exactly.

---

## 4. AdamW, and what weight decay touches

AdamW keeps two running averages per weight, the gradient's recent mean and its recent squared
size, and scales each weight's step by them. Weights with consistently large gradients get
proportionally smaller steps, so one global lr works for all 10.6M weights. Those two averages are
why AdamW needs 8 extra bytes per parameter, and why they must be in every checkpoint.

**Weight decay** (0.1) shrinks weights slightly towards zero each step, discouraging solutions that
depend on a few huge weights. It applies to the weight matrices and embedding, and **not** to
RMSNorm gains: those set the scale of each layer's input, and pulling them towards zero would fight
that job (`train/optim.py`).

`beta2 = 0.99` in the `ref` experiment (the default is 0.95) makes the squared-gradient average
remember **more** steps: roughly 1 / (1 − beta2), so ~100 steps instead of ~20. nanoGPT uses it for
this run because each step sees only 16,384 tokens, so single-step gradient sizes are noisy, and a
longer average smooths that out.

---

## 5. Evaluation: validation loss and samples

Every `eval_every_steps` (250 for `ref`), the trainer:

1. **Estimates validation and training loss** on `eval_iters` batches each, drawn from a fixed seed
   so every eval scores exactly the same windows and numbers are comparable over time. Dropout is off
   (`model.eval()`).
2. **Prints the gap** (`val − train`). Early on it is near zero. A widening gap is the signature of
   memorization, and for Shakespeare it is expected in the second half of the run.
3. **Generates samples** from the project's prompts (`ROMEO:\n`, …), also from a fixed seed, so a
   sample at step 1,000 and one at step 2,000 differ only because the model changed.
4. **Saves `ckpt/best/`** if validation loss improved.

**Why samples at every eval** (CONTRIBUTING.md makes this non-negotiable): loss is a single number that
can improve for bad reasons. Leakage, a broken mask, or memorization can all produce a great loss
with garbage or copied output. Reading ten lines of generated text catches what the number hides.

---

## 6. Checkpoints and resume: stopping is normal

The machine is switched off most nights, so **every real run is a resumed run**. A checkpoint
therefore holds everything needed to continue as if nothing happened:

| Saved | Why it is needed |
|---|---|
| Model weights | obviously |
| AdamW's two averages | without them, the first steps after resume act like a cold start and the loss curve jumps |
| `step`, `tokens_seen` | where in the data and the lr schedule the run is |
| Train sampler state | so the next batches are the ones an unbroken run would have drawn |
| All four RNG states (Python, NumPy, PyTorch CPU, PyTorch CUDA) | dropout draws random numbers every step; restoring them replays the same dropout masks |
| GPU-seconds used, best validation loss, measured tokens/s | so `slm runs list` and the ETA stay correct across sessions |

**How much does each piece matter?** The resume-equivalence test trains 20 steps straight, and
separately 10 steps, stop, a fresh `Trainer` resumes, 10 more steps. On the CPU the two loss curves
match **exactly (difference 0.0)**. Its companion test breaks just one piece, skipping the RNG
restore, and the curves diverge by up to **0.046** within a few steps. A resume that is "almost
right" produces a run that is subtly different from the one you configured, and nothing errors.

**Atomic writes.** A checkpoint is written to `ckpt/step_0002499.tmp/`, fsynced, then renamed. A
power cut mid-save leaves the previous checkpoint intact plus a `.tmp` directory that resume ignores
and the next save clears. We saw exactly that happen during testing (§7). The last 3 step
checkpoints are kept; older ones are pruned.

**When checkpoints happen:** every 20 minutes of wall-clock time (time, not steps, because "how much
work would a power cut lose" is a question about time), on Ctrl-C, at a session limit
(`--max-minutes`, `--max-steps`), and at the end.

**Resume is not a flag.** `slm pretrain shakespeare_char/ref` finds its run directory from a hash of
the configuration and data (`train/run.py`), sees a checkpoint, and continues. Settings that only
change how the run is *watched* (eval cadence, checkpoint interval, compile, tracker) are left out
of that hash, so you can change them between sessions without starting over.

---

## 7. Ctrl-C, and a bug worth knowing about

Pressing Ctrl-C finishes the current step, saves a checkpoint and exits:

```
SIGINT: finishing this step, then checkpointing. Press Ctrl-C again to abort without saving.
  checkpoint  ckpt/step_0003387  (0.2s)
stopped (SIGINT) at step 3387, 67.7% done. Run the same command to resume.
```

The first version of this had a real bug. In a terminal, Ctrl-C sends SIGINT to the whole
foreground **process group**. Under `uv run slm …` that group holds both `uv` and the Python
process, and `uv` *also forwards* the signal to Python. So one keypress arrived **twice**, a few
milliseconds apart, and the handler read the second as "press Ctrl-C again to abort without saving".
Every Ctrl-C threw the checkpoint away mid-write, leaving a `.tmp` directory behind.

It was found by sending a signal to the process group from a test harness, and confirmed with a
10-line script that counts SIGINTs (it received two). The fix: a repeat signal within one second of
the first counts as the same keypress. A deliberate second press later still aborts.

Two things remain visible and are harmless:

- `torch.compile` runs worker processes in the same group, and they may print a `KeyboardInterrupt`
  traceback as they exit. The trainer's own lines (`checkpoint …`, `stopped …`) are what count.
- `tmux` keeps a run alive when the terminal closes. It does **not** survive a power-off; the 20-minute
  checkpoint is what survives that.

---

## 8. Throughput, MFU and GPU-hours

At step 50 the trainer replaces its estimate with a measurement:

```
  measured    808,923 tokens/s  ·  57.4 TFLOPS  ·  MFU 51.2% of 112.1 (measured by slm doctor --bench)
              memory 1.58 GiB peak  ·  remaining 1.7 GPU-min (measured)
```

- **TFLOPS achieved** = tokens/s × FLOPs per token (71.0M for `ref`; `slm model` prints it).
- **MFU** = that ÷ the card's measured bf16 peak. Steps 1–10 are excluded: they include
  `torch.compile`'s compilation.
- **GPU-hours remaining** uses measured tokens/s from then on, and is saved in the checkpoint so the
  number is right after a resume too.

Measured MFU per preset (Shakespeare, batch 64 × 256, compiled, bf16):

| Preset | tokens/s | TFLOPS | MFU | Peak memory |
|---|---|---|---|---|
| `nano` (0.85M) | 3,915,615 | 26.4 | 23.6% | 0.35 GiB |
| `micro` (4.8M) | 1,378,216 | 46.5 | 41.5% | 0.97 GiB |
| `ref` (10.6M) | 808,923 | 57.4 | 51.2% | 1.58 GiB |
| `tiny` (25.7M) | 360,021 | 60.1 | 53.6% | 2.72 GiB |

**This is much better than the design expected.** DESIGN predicted 10–20% MFU at the small presets
because small matrices don't keep tensor cores busy. `torch.compile` fusing the small operations,
plus 16,384 tokens per step, closes most of that gap. From `micro` upwards the real throughput
**beats** the 43 TFLOPS planning figure, so budgets based on it are conservative. Below that, `nano`
is genuinely launch-bound. Memory stays under 3 GiB of 16: compute, not memory, is the constraint,
as DESIGN said.

GPU-hours *used* counts wall-clock time the trainer was running, evals and checkpoints included,
and adds up across sessions. The whole `ref` run is ~82M tokens ÷ ~810K tokens/s ≈ **1.7 GPU-minutes**
of training, plus evals.

---

## 9. What a run leaves behind

```
$SLM_HOME/runs/run-a1d5f0a224ff/
├── manifest.json          what went in: packed data + tokenizer IDs, the full config
├── config.resolved.yaml   the exact merged config
├── status.json            progress, GPU-hours, best val, last checkpoint (what `slm runs list` reads)
├── metrics.jsonl          one JSON line per log step and per eval (samples included)
├── train.log              everything printed, flushed line by line, survives a power-off
├── tb/                    TensorBoard event files
└── ckpt/
    ├── step_0001452/      model.pt  optimizer.pt  state.pt
    ├── step_0002499/
    ├── step_0003387/      ← the last 3 are kept
    └── best/              ← lowest validation loss so far
```

`metrics.jsonl` and `status.json` exist so that no tracking tool is ever required to know how a run
went. TensorBoard adds the curves: `uv run tensorboard --logdir $SLM_HOME/runs`, then open
http://localhost:6006 in a Windows browser (WSL forwards localhost).

---

## 10. How we know it is right

| Claim | Test |
|---|---|
| Stop + resume = never stopped (exact, on CPU) | `test_resume_is_equivalent_to_never_stopping` |
| …and the test would notice a broken resume | `test_resume_test_is_sensitive` (drift 0.046) |
| Loss falls; samples printed at step 0 and at every eval | `test_loss_falls_and_samples_are_printed` |
| A finished run is not retrained | `test_completed_run_is_not_retrained` |
| Last N checkpoints kept, `best` kept, half-written ones ignored | `test_checkpoints_are_pruned_and_best_is_kept`, `test_half_written_checkpoint_is_ignored` |
| Watching settings keep the run; learning settings start a new one | `test_run_identity` |
| Warmup, cosine midpoint and floor are exact | `test_warmup_is_linear_from_near_zero`, `test_cosine_peak_midpoint_and_floor` |
| NaN steps skipped, then abort | `test_guard_skips_then_aborts` |
| 50 compiled bf16 steps on the GPU learn | `tests/gpu/test_train_gpu.py` |

**Rejected:** a training framework (HF `Trainer`, Lightning, Accelerate). Each would provide this
loop in one line, and hide every decision on this page. slmkit exists to make those decisions
visible (CONTRIBUTING.md rule 6).
