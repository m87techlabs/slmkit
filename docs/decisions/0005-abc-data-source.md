# ADR 0005 — ABC data: public-domain tune books, not The Session

**Status:** accepted (M2, Phase A)

## Context

DESIGN §5.1 planned `abc_music` around the thesession.org data dump (`adactio/TheSession-data`,
~45K tunes). Checking its licence before writing the ingest found an explicit clause:

> You may not use, adapt, modify, or process the material in any way with Large Language Models.
> This includes but is not limited to training Large Language Models […]

The clause restricts *use*, not only redistribution, so training on the dump privately would
still be outside the licence. slmkit is also intended to be public, with runbooks, samples and a
model card that name the training data. The only exception in the licence is for accessibility
tools, which slmkit is not.

The tunes themselves are mostly traditional and in the public domain. What the licence covers is
The Session's collection of community transcriptions.

## Decision

Use the ABC transcriptions of three **public-domain tune books** that ship inside the `music21`
package (BSD-3-Clause), read from the version pinned by `uv.lock`:

| Collection | Published | Tunes | Transcribed by (from the files' `Z:` lines) |
|---|---|---|---|
| Ryan's Mammoth Collection | 1883 | 1,059 | Ray Davies, Bob Puckette |
| O'Neill's Music of Ireland ("1850") | 1903 | 2,009 | Henrik Norbeck, Bob Safranek, Trish O'Neil, others |
| Aird's Selection of Airs | 1778–1803 | 1,180 | Jack Campin (2009) |

The Essen folksong collection, also in `music21`, is excluded: its licence file describes its
status as "unclear", with permission given only for distribution in music21. The Nottingham
Music Database is excluded: its cleaned version is GPL-3.0, and the original's terms are unstated.

Transcriber credit lines (`Z:`), including their email addresses, are stripped from the training
text and credited in `projects/abc_music/README.md` instead.

## Consequences

- **Smaller corpus:** 4,248 tunes and ~1.1M characters after cleaning, instead of ~45K tunes.
  Transposition augmentation brings the training split to 2.85M tokens. That suits the 1–5M
  parameter models DESIGN already planned for ABC. M2's goal is to exercise the full lifecycle,
  not to build the best possible folk-tune model.
- **Patchier labels:** `R:` (rhythm) is on 99% of Ryan's, 44% of O'Neill's and ~0% of Aird's
  tunes. SFT prompts must work from meter and key, and use rhythm only when present.
- **Duplicates across books:** the same tune can appear in several books and as several
  "settings". Grouping uses a melody fingerprint plus distinctive titles instead of a tune ID.
- **Residual licence risk:** the source books are public domain, but the transcription files state
  no licence of their own. Faithful transcriptions of public-domain scores are generally not
  treated as new creative work, but that is an assumption, not a grant. If a transcriber or
  rights-holder objects, the affected collection is removed and the dataset rebuilt (a changed
  ingest produces new artifact IDs, so nothing stale survives).
- No download at ingest: the data comes from an installed, version-pinned package.

## Alternatives considered

- **Use The Session privately anyway:** rejected. The licence restricts use, not publication.
- **Ask The Session for permission:** legitimate, but not pursued for now. If granted, it would
  become an additional collection in the same project, with the permission recorded here.
- **Switch M2 to chess (Lichess CC0):** the cleanest licence available, but it pulls M3's
  fixed-vocabulary tokenizer and streaming ingest forward, and chess has no natural SFT task.
  Rejected to keep the milestone order.
