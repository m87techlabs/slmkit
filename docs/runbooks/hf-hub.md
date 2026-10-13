# Runbook: publishing on the Hugging Face Hub

<!-- slm-studio: projects=abc_music -->

*How an export becomes a public model that anyone can load with `transformers`, and a live demo
anyone can try in a browser. The steps are generic; the worked example is `abc-folk:3`. Concepts:
[`../concepts/serving.md`](../concepts/serving.md) §9 and
[`../concepts/browser-inference.md`](../concepts/browser-inference.md). Decisions:
[ADR 0010](../decisions/0010-publishing-to-the-hub.md), [ADR 0011](../decisions/0011-browser-inference.md). Roadmap: the "Hugging Face Hub" section of
[`../ROADMAP.md`](../ROADMAP.md).*

| | Where | What |
|---|---|---|
| **Model** | [huggingface.co/m87techlabs/abc-folk](https://huggingface.co/m87techlabs/abc-folk) | the export directory as is, plus `README.md` (a copy of `MODEL_CARD.md`). Tag `v3` = slmkit's `abc-folk:3` |
| **Space** | [huggingface.co/spaces/m87techlabs/abc-folk](https://huggingface.co/spaces/m87techlabs/abc-folk) | a **static** Space: the site `slm site abc-folk:3` builds. The model runs in each visitor's browser; there is no server |

Publishing is the one step in slmkit that is **outward-facing and hard to take back**: once public,
files can be copied. Every step here is run by a person, and nothing in slmkit publishes on its own.

Infrastructure analogy: the Hub is a container registry for model files, each repository a git repo.
A model version is an immutable image tag; a Space is a small hosted deployment of one image.

---

## H.0 What publishing needs from slmkit

Nothing new at run time. Two things were added so that an export can be uploaded as is (ADR 0010):

- **The model card opens with the Hub's metadata**, a YAML block with licence, library, task and
  tags. Export code version 4 and later (runbook M2 §E.7).
- **The project states the licence and the data's status** through `Project.model_card_info()`. Only
  the project can know whether its weights can be licensed. `abc_music` says Apache-2.0 and states the
  open question about its transcriptions (ADR 0005), with a promise to withdraw the model if anyone
  objects.

The `hf` command-line client is run through `uvx`, so it isn't a slmkit dependency (STACK.md §7).

---

## H.1 Once: account, organisation, token, CLI

1. Create a Hugging Face account, then an organisation if the models should live under a team name
   (https://huggingface.co/organizations/new).
2. Create an access token with write access: Settings → Access Tokens. A classic **Write** token
   covers every organisation you belong to. A **fine-grained** token needs write access granted per
   organisation.
3. Log in:

```bash
alias hf='uvx --from huggingface_hub==2.2.0 hf'
hf auth login          # paste the token; "Add token as git credential?" → n
hf auth whoami
```

`whoami` must list the organisation under `orgs`, **by its username**.

**Pitfalls met on the way** (each cost a round trip):

- **An organisation has two names.** The *display name* is shown on its page; the *username* is the
  namespace in every URL and command. They can differ: an organisation shown as "m87techlabs" was
  first created with the username `m87admin-org`. `hf repos create m87techlabs/abc-folk` then failed
  with `403 Forbidden: You don't have the rights to create a model under the namespace`. Fix: rename
  the username under the organisation's Settings → Account, while it is still empty.
- **Password managers fill settings forms.** On the account settings page, a password manager filled
  the *Username* field with an email address. Saving would have tried to rename the account. Check
  every field before pressing Save there.
- **A 403 after that is the token**: a fine-grained token without write access to that organisation.
- **Docker Spaces aren't free any more.** Creating a Docker Space returned `402 Payment Required`,
  under the organisation (`… requires a Team or Enterprise plan for organization m87techlabs`) and
  under the personal account (`… requires a PRO subscription`). Static Spaces, which serve files only,
  are free for everyone. That is why the demo became a static site with the model running in the
  browser (ADR 0011, H.5).

---

## H.2 Export with the Hub metadata

Use an export made with code version 4 or later, so its card carries the metadata. `abc-folk:3` is
the fine-tuned run from M2, exported again for this (runbook M2 §E.7):

```bash
head -4 ~/slm/models/abc-folk/3/MODEL_CARD.md
```

```
---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
```

Before publishing, look for anything private in the files. Exports record run IDs and settings, not
paths, but check:

```bash
grep -rIl "$HOME" ~/slm/models/abc-folk/3/ || echo "no local paths"
```

```
no local paths
```

---

## H.3 Upload the model, its card and its version tag

```bash
hf repos create m87techlabs/abc-folk --type model
hf upload m87techlabs/abc-folk ~/slm/models/abc-folk/3 . \
  --commit-message "abc-folk:3, exported by slmkit from run-9b890db6dced"
hf upload m87techlabs/abc-folk ~/slm/models/abc-folk/3/MODEL_CARD.md README.md \
  --commit-message "Model card"
hf repos tag create m87techlabs/abc-folk v3
```

**Order matters:** tag *after* the README upload, so `v3` includes the card. The tag is what makes the
Hub version match slmkit's: the repository's `main` moves on when `abc-folk:4` is uploaded, but `v3`
keeps pointing at these files, and the Space downloads `v3`.

**Why a copy, not a rename.** The Hub shows `README.md` as the model page. Exports keep the name
`MODEL_CARD.md`, which existing exports, docs and runbook output already use, so the upload copies it.

---

## H.4 Verify the model, as a stranger would

These checks use the Hub's public API and no token, so they see what anyone sees.

**The page has its metadata:**

```bash
curl -s https://huggingface.co/api/models/m87techlabs/abc-folk | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['id'], 'private' if d['private'] else 'public')
print('licence:', d['cardData']['license'], '| library:', d['library_name'], '| task:', d['pipeline_tag'])
print('tags:', ', '.join(d['cardData']['tags']))"
```

```
m87techlabs/abc-folk public
licence: apache-2.0 | library: transformers | task: text-generation
tags: slmkit, llama, from-scratch, abc-music, music, abc-notation, folk-music, symbolic-music
```

Before the README upload, the same call gave `licence` and `task` as missing: the Hub reads metadata
only from `README.md`.

**The tag exists and points at the right commit:**

```bash
curl -s https://huggingface.co/api/models/m87techlabs/abc-folk/refs | python3 -c "
import json, sys
d = json.load(sys.stdin)
for r in d['branches'] + d['tags']: print(r['ref'], r['targetCommit'][:12])"
```

```
refs/heads/main 90441cb5f36e
refs/tags/v3 90441cb5f36e
```

**The published weights are byte-for-byte the export.** The Hub stores large files with their
SHA-256, so compare it with the local file:

```bash
curl -s https://huggingface.co/api/models/m87techlabs/abc-folk/tree/v3 | python3 -c "
import json, sys
print(next(f['lfs']['oid'] for f in json.load(sys.stdin) if f['path'] == 'model.safetensors'), '(Hub, v3)')"
sha256sum ~/slm/models/abc-folk/3/model.safetensors | cut -d' ' -f1 | sed 's/$/ (local)/'
```

```
0ff22be8daed1a48fc80205f4fe484774a95e2f23762e2e9ed86b2cb99024af9 (Hub, v3)
0ff22be8daed1a48fc80205f4fe484774a95e2f23762e2e9ed86b2cb99024af9 (local)
```

**It loads from the Hub with `transformers`**, no slmkit involved. `HF_TOKEN=` and an empty cache make
it an anonymous download:

```bash
HF_TOKEN= HF_HUB_CACHE=$(mktemp -d) uv run --extra export python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('m87techlabs/abc-folk', revision='v3')
model = AutoModelForCausalLM.from_pretrained('m87techlabs/abc-folk', revision='v3')
ids = tok('% a jig in G major, 6/8 time\n', return_tensors='pt').input_ids
print(model.num_parameters(), tok.decode(model.generate(ids, max_new_tokens=40, do_sample=False)[0]))
"
```

```
864256 % a jig in G major, 6/8 time
R:jig
M:6/8
L:1/8
K:G
(G/A/)|BGG G2 G|BG
```

(After a "Loading weights" progress bar.) Greedy decoding, so this is exact: the same characters as
the local copy gives.

---

## H.5 The demo: a static Space, the model running in the browser

**What.** `slm site abc-folk:3` builds a folder of static files: the playground page from runbook M2
§E.6, `model.js` (the same network as `llama.py`, in JavaScript), the model's files and the folk-tune
viewer. A static Space serves the folder as it is, and each visitor's browser runs the model.
`../concepts/browser-inference.md` explains how and why.

**Build the site.** It is an artifact like any other, so building it again is a no-op:

```bash
uv run slm site abc-folk:3
ls ~/slm/sites/site-5c053f5e701f ~/slm/sites/site-5c053f5e701f/model
```

```
abc-folk:3: site already built at /home/you/slm/sites/site-5c053f5e701f
try it: python3 -m http.server -d /home/you/slm/sites/site-5c053f5e701f 8000   then open http://localhost:8000/
/home/you/slm/sites/site-5c053f5e701f:
engine-worker.js
index.html
info
manifest.json
model
model.js
ui

/home/you/slm/sites/site-5c053f5e701f/model:
config.json
generation_config.json
model.safetensors
tokenizer.json
```

(The first build prints `built site-5c053f5e701f for abc-folk:3 -> …`.) Try it locally with the
`http.server` line it prints: the page header ends in "· your browser", and Generate needs no server
beyond the one handing out files.

**It is the same model.** The parity tests run `model.js` in headless Chrome against PyTorch (skipped if
no Chrome or Chromium is found; set `SLM_CHROME` to point at one):

```bash
uv run pytest -q tests/unit/test_site.py
```

```
6 passed
```

Measured by hand on abc-folk:3: the largest logit difference over a full prompt is 2.4e-6, greedy output
is the same text as `transformers` (H.4) and `slm serve`, and a 189-token jig takes 0.12 s in Chrome
(about 1,500 tokens/s).

**Publish it.** Create a static Space, upload the site, then a `README.md` whose YAML header says
`sdk: static` and `app_file: index.html` and names the model (`models: [m87techlabs/abc-folk]`), which
links the Space and the model page to each other:

```bash
hf repos create m87techlabs/abc-folk --type space --sdk static
hf upload m87techlabs/abc-folk ~/slm/sites/site-5c053f5e701f . --type space \
  --commit-message "slm site abc-folk:3"
hf upload m87techlabs/abc-folk <the Space's README.md> README.md --type space \
  --commit-message "Space card"
```

A model and a Space can share a name: they are different repository types. A static Space has no build
step; it is live within a minute.

**Why it's built this way.**

- **No server:** a Docker Space needs a paid plan (H.1), and a server is more than a 3.5 MB model needs.
  The browser generates faster than `slm serve` on a CPU, because at this size PyTorch's per-operation
  overhead dominates.
- **The site carries its own copy of the model files** rather than fetching them from the model
  repository: it works on any static host, and its parity was tested on exactly those bytes.
- **The same page as `slm serve`:** one playground to maintain, and project viewers work in both.
- **Rejected:** a Gradio app (a second interface and a framework slmkit doesn't use);
  transformers.js with an ONNX export (a second model format standing in for the part slmkit writes
  itself). See ADR 0011.

---

## H.6 Updating, and taking it down

- **A new version** (`abc-folk:4`): upload its directory and card to the same model repository and
  tag it `v4`. Then `slm site abc-folk:4`, and upload that site to the Space. `v3` stays downloadable
  from the model repository, and the Space's git history keeps the old site.
- **Withdrawal**, if a rights-holder objects to the data (ADR 0005), is one command per repository:
  `hf repos delete m87techlabs/abc-folk --type space` and `hf repos delete m87techlabs/abc-folk`.

---

## Done when

- [x] `abc-folk:3` is public on the Hub with its card, licence, task and tags, and tag `v3` (H.3, H.4).
- [x] The Hub's weights are byte-identical to the export, and `transformers` loads them anonymously and
      generates the expected tune (H.4).
- [x] `slm site` builds the static site, and `model.js` matches PyTorch in Chrome (H.5).
- [ ] The static Space is live; the playground draws and plays a tune there (H.5).
- [ ] The README links to the Space.
- [ ] **You** have run H.4 and opened both pages.
