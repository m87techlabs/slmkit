# Two models, side by side: Shakespeare and folk tunes

*No music background needed. Hands-on: `slm studio` → Playground, or
[`../runbooks/m2-abc-music.md`](../runbooks/m2-abc-music.md) §E.6. Terms:
[`../GLOSSARY.md`](../GLOSSARY.md).*

slmkit has trained two kinds of model. They are the **same machine**, the same code and the same kind
of network, pointed at different text. What differs is what they read, and how you can tell whether
what they write is any good.

---

## 1. The same machine, two kinds of text

Both are **next-character predictors**. Given the text so far, the model scores every character it
knows and picks one; then it does it again with that character added. That's all either model ever
does. Everything else is what they were shown while learning.

| | M1: `shakespeare_char` | M2: `abc_music` |
|---|---|---|
| **Read** | ~1 MB of Shakespeare's plays | 4,248 folk tunes from three 18th–20th-century tune books |
| **Vocabulary** | 65 characters | 87 characters |
| **Model** | `ref` size: 10.6M parameters | `nano` size: 0.86M (and `micro`, 4.8M) |
| **You give it** | a speaker's name: `ROMEO:` | a request in words: `% a jig in G major, 6/8 time` |
| **It writes** | a speech, in Shakespeare's style | a tune, written as text in **ABC notation** |
| **Is the output good?** | you can tell by reading it | you need to read music, or a program that does |
| **Graded by** | validation loss only | loss, plus programs that check the music (§3) |
| **Purpose** | prove the engine trains correctly (M1) | prove the whole lifecycle on a real task (M2) |

What the Shakespeare model writes, asked to continue `JULIET:` / `O Romeo, Romeo,` (seed 2):

```
JULIET:
O Romeo, Romeo, that cohangeth me to his act
Of God and first that company thee to the bawd.
And his good words, here is God lost and me.
```

It has learned the *look* of Shakespeare: speaker names, verse lines, old words ("thee", "bawd"),
punctuation. It has no idea what any of it means, and invents words ("cohangeth"), because it works one
character at a time and has read only 1 MB. That's why M1's measure of success was a number (validation
loss 1.29, matching the published reference), not the quality of the prose.

---

## 2. What M2 set out to prove, and why music

M1 proved the engine *trains*. M2 had to prove the whole lifecycle works on a real task:

1. **prepare data** honestly (no tune in both training and validation, augmentation only on training),
2. **choose a tokenizer** and compare it fairly,
3. **pretrain**, then **fine-tune** so the model answers requests in plain words,
4. **evaluate automatically**, with programs instead of people,
5. **export** to a standard format, and **serve** it to a browser.

Step 4 is the hard part with prose: a program can't tell whether a paragraph is good. **Music in ABC
notation solves that.** It's plain text, so the same character model learns it, and a program can check
it objectively: does it play, does every bar have the right length, does it end where tunes usually end.
That's why slmkit only picks projects with a free, automatic grader (chess is next for the same reason:
a program knows which moves are legal).

So the point of M2 isn't great music. It's a model you built from nothing, which takes a request in
English and answers with something a program can **verify**, plus the measurements to show how often
it succeeds.

---

## 3. Reading a generated tune, without reading music

A real answer from `abc-folk:2` to `% a jig in G major, 6/8 time` (seed 3, on the GPU):

```
% a jig in G major, 6/8 time        ← your request (a comment line)
R:jig                               ← rhythm: what kind of tune
M:6/8                               ← time signature: what fits in one bar
L:1/8                               ← unit note: a plain letter lasts 1/8
K:G                                 ← key: G major, so the home note is G
uD \                                ← a lead-in note (u = "up-bow", a fiddle mark)
| G2B cde | f2d f2d | d2d efg | fdd f2d |
d2B cde | fed e2d | cde fdB | c2d c2d \
| ddd f2d | gfg f2d | e2d edc \
| d2d cde | d2d fc'd' | b2d dBd | edc d2||
```

**The headers answer your request.** You asked for a jig, in G major, in 6/8, and the four header lines
say exactly that. Checking this is the simplest test of whether fine-tuning worked: the base model,
asked the same thing in words, gets the bars right only about a quarter of the time (sft.md §4).

**The letters are notes.** `C D E F G A B` are the white keys of a piano; lowercase is an octave higher;
`'` is higher still. A number makes a note longer: with `L:1/8`, `G2` lasts two eighths and `B` one.

**`|` separates bars, and each bar must add up.** 6/8 means six eighth-notes per bar. Take the first bar,
`G2B cde`: 2 + 1 + 1 + 1 + 1 = 6. ✓ Every bar here adds up. The last one, `edc d2`, has 5, and the lead-in
`D` at the very start has 1: together a full bar, the standard way to start a tune on an upbeat (a
*pickup*). The **bar accuracy** grader knows this convention, and scores this tune 1.0.

**It plays.** `abc2midi`, the reference ABC player, reads it with no errors: the **plays** grader scores 1.0.

**It doesn't end at home.** The last note is `d` (D). In G major the home note is G, and about 80% of real
tunes end there. D is a common ending in real tunes too, but the **ends-on-tonic** grader asks for G, and
scores this tune 0.

That is the whole evaluation, done by programs over 600 tunes per model instead of by eye over one:

| Grader | Asks, in plain words | Real tunes | Best model |
|---|---|---|---|
| plays | does a music program read it without errors? | ~100% | 74% |
| bar accuracy | does every bar hold the right number of beats? | 99% | 81% |
| ends on tonic | does it end on the key's home note? | 80% | 42% |
| novelty | is it new, rather than copied from the training tunes? | — | 99.8% |

The playground's **Reading this tune** panel does this reading for every tune you generate: it compares
what you asked for with what it wrote, explains each header, and says where it ends.

---

## 4. Try it yourself

Start the studio (`uv run slm studio start`), open **Playground**, and pick a model. Every playground has
**Try:** buttons with the prompts the evaluation used, and a box you can type into.

**Shakespeare** (project `shakespeare_char`, model `shakespeare:1`). Type the start of a speech and let
it continue:

| Type | What happens |
|---|---|
| `ROMEO:` then Enter | a speech, then other characters answer |
| `KING RICHARD III:` then a line break and `Now is the` | it carries on in verse |
| `First Citizen:` then a line break and `We are` | a crowd scene |
| `Hello, how are you?` | it continues as if it were a line in a play; it can't chat |

**Folk tunes** (project `abc_music`, model `abc-folk:2`). Use **Ask for … in the key of …** to build a
request, or type one in the same form:

| Type | What happens |
|---|---|
| `% a reel in D major, 2/2 time` | a fast dance tune; watch "you asked for / it wrote" |
| `% a hornpipe in A major, 2/4 time` | a different rhythm, a different key |
| `% a tune in 3/4 time in E minor` | a slower melody, in a minor (sadder) key |
| `% a reel in D major, 4/4 time` | it usually writes 2/4 instead: 4/4 reels barely exist in its tune books, so a request can't override what it never saw (evaluation.md §6) |
| `% Write me a happy song?` | the page warns that `W` and `?` aren't in its 87-character vocabulary; it learned only a few fixed phrasings |

Same seed, same tune; change the temperature to see how much the model is allowed to wander (0.3: safe and
repetitive; 1.2: adventurous and often broken).

---

## 5. What neither model can do

- **Understand anything.** Both reproduce patterns of characters. The Shakespeare model has no idea who
  Romeo is; the tune model has never heard a note.
- **Follow open-ended instructions.** The tune model learned eight phrasings built from a few words. That
  it answers those reliably is the result: fine-tuning taught it an interface (sft.md §5).
- **Scale up for free.** Bigger models and more data help (experiments.md §4), and cost GPU-hours
  ([`Parameters` in the studio](../runbooks/studio.md) estimates how many).
