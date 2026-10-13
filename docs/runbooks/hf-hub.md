# Runbook: publishing on the Hugging Face Hub

<!-- slm-studio: projects=abc_music -->

*How an export becomes a public model that anyone can load with `transformers`, and a live demo
anyone can try in a browser. The steps are generic; the worked example is `abc-folk:3`. Concepts:
[`../concepts/serving.md`](../concepts/serving.md) §9. Decision:
[ADR 0010](../decisions/0010-publishing-to-the-hub.md). Roadmap: the "Hugging Face Hub" section of
[`../ROADMAP.md`](../ROADMAP.md).*

| | Where | What |
|---|---|---|
| **Model** | [huggingface.co/m87techlabs/abc-folk](https://huggingface.co/m87techlabs/abc-folk) | the export directory as is, plus `README.md` (a copy of `MODEL_CARD.md`). Tag `v3` = slmkit's `abc-folk:3` |
| **Space** | [huggingface.co/spaces/m87admin/abc-folk](https://huggingface.co/spaces/m87admin/abc-folk) | a Dockerfile that installs a released slmkit and runs `slm serve` on the model's `v3` tag |

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
- **Organisations pay for Docker Spaces.** Creating the Space under the organisation returned
  `402 Payment Required: … hosting Gradio and Docker Spaces on free cpu-basic requires a Team or
  Enterprise plan for organization m87techlabs`. Static Spaces (HTML only) stay free for organisations,
  and personal accounts can still run a Docker Space on the free CPU tier. So the Space lives under the
  personal account `m87admin` and serves the organisation's model (H.5).

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

## H.5 The demo: a Docker Space running `slm serve`

**What.** A Space is a demo app built from its own git repository. A *Docker* Space builds the
repository's Dockerfile and sends visitors to one port. slmkit's Space is `slm serve`, unchanged: the
same playground page as runbook M2 §E.6, viewer and all, on the free CPU tier (2 vCPUs, 16 GB).

The Space repository holds two files. The Dockerfile:

```
FROM python:3.12-slim
RUN useradd -m -u 1000 user            # Spaces run containers as user 1000
USER user
ENV HOME=/home/user PATH=/home/user/venv/bin:$PATH SLM_HOME=/home/user/slm
WORKDIR /home/user
RUN python -m venv /home/user/venv
# CPU-only PyTorch first, the version in uv.lock: slmkit's torch>=2.7 is then satisfied,
# and pip doesn't fetch the multi-GB CUDA build.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
ARG SLMKIT_REF=v0.1.0
RUN pip install "slmkit @ https://github.com/m87techlabs/slmkit/archive/refs/tags/${SLMKIT_REF}.tar.gz" \
    "huggingface_hub==2.2.0"
ARG MODEL_REPO=m87techlabs/abc-folk
ARG MODEL_REV=v3
RUN hf download "${MODEL_REPO}" --revision "${MODEL_REV}" --local-dir "${SLM_HOME}/models/abc-folk/3"
EXPOSE 7860
CMD ["slm", "serve", "--model", "abc-folk:3", "--host", "0.0.0.0", "--port", "7860", "--threads", "2"]
```

(shortened here; the full file, with comments, is in the Space repository). The other file is a
`README.md` whose YAML header sets `sdk: docker`, `app_port: 7860` and `models: [m87techlabs/abc-folk]`,
which links the model page and the Space to each other.

**How.**

```bash
hf repos create m87admin/abc-folk --type space --sdk docker
hf upload m87admin/abc-folk <dir with Dockerfile and README.md> . --type space \
  --commit-message "slm serve (slmkit v0.1.0) for abc-folk:3"
```

It is created under the personal account: organisations need a paid plan for Docker Spaces (H.1).
The Space's README still names the organisation's model, so the two pages link to each other. The first build takes a
few minutes; its log is at `https://huggingface.co/spaces/m87admin/abc-folk?logs=build`.

**Verify.** The recipe was first run by hand in a clean virtual environment (CPU PyTorch 2.11.0, slmkit
from the v0.1.0 tarball, `slm serve --model abc-folk:3`): one jig of 160 tokens in 0.36 s, 446 tokens/s.
Against the live Space:

```bash
curl -s https://m87admin-abc-folk.hf.space/health
```

**Why it's built this way.**

- **`slm serve`, not a Gradio app** (the Hub's default): a second interface would need keeping in step
  with the first, and Gradio is a framework slmkit doesn't use (CONTRIBUTING.md rule 6).
- **A released slmkit and a tagged model**, both pinned: rebuilding the Space later gives the same
  thing, the way a pinned base image does.
- **CPU-only PyTorch, installed first.** The default PyTorch from PyPI bundles CUDA libraries a CPU
  Space can't use.
- **The Dockerfile lives in the Space's repository, not slmkit's.** It deploys one model of one
  project, and container design is M4's work (ADR 0010).

---

## H.6 Updating, and taking it down

- **A new version** (`abc-folk:4`): upload its directory and card to the same model repository, tag
  it `v4`, then change `MODEL_REV` and the model version in the Space's Dockerfile and upload that.
  `v3` stays downloadable.
- **Withdrawal**, if a rights-holder objects to the data (ADR 0005), is one command per repository:
  `hf repos delete m87admin/abc-folk --type space` and `hf repos delete m87techlabs/abc-folk`.

---

## Done when

- [x] `abc-folk:3` is public on the Hub with its card, licence, task and tags, and tag `v3` (H.3, H.4).
- [x] The Hub's weights are byte-identical to the export, and `transformers` loads them anonymously and
      generates the expected tune (H.4).
- [ ] The Space builds and answers `/health`; the playground draws and plays a tune (H.5).
- [ ] The README links to the Space.
- [ ] **You** have run H.4 and opened both pages.
