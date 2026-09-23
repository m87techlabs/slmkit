"""The data stages: ingest -> prepare -> tokenize -> pack.

Each `ensure_*` function computes its artifact ID from config and upstream IDs, returns
immediately if that artifact already exists, and otherwise builds it (building any missing
upstream artifacts first). That makes every stage idempotent and `slm run` safe to re-invoke
after a week away: it is `make` with content hashes instead of timestamps.

`CODE_VERSION` is hashed into every artifact ID. Bump a stage's number when a change to its
code would change its output; leave it alone for refactors. That is what invalidates stale
artifacts without invalidating everything on every commit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from slmkit import artifacts
from slmkit.config.load import Experiment
from slmkit.data.pack import DTYPE, pack_split
from slmkit.data.split import SPLITS, read_docs, split_documents, write_docs
from slmkit.project_api import Project
from slmkit.registry import load_project
from slmkit.tokenizers import CharTokenizer, Tokenizer, load_tokenizer

CODE_VERSION = {"raw": 1, "dataset": 1, "tokenizer": 1, "packed": 1}

Log = Callable[[str], None]


@dataclass(frozen=True)
class StageResult:
    kind: str
    id: str
    path: Path
    skipped: bool


def _noop(_: str) -> None:
    pass


def _report(log: Log, result: StageResult) -> StageResult:
    verb = "exists " if result.skipped else "built  "
    log(f"  {result.kind:<9} {verb} {result.id}  {result.path}")
    return result


def project_for(exp: Experiment) -> Project:
    return load_project(exp.project, exp.config.project.args, artifacts.slm_home())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- ingest


def ensure_raw(exp: Experiment, log: Log = _noop) -> StageResult:
    """Download source data to $SLM_HOME/raw/<project>/. Immutable once written."""
    final = artifacts.slm_home() / "raw" / exp.project
    if (final / artifacts.MANIFEST).is_file():
        manifest = artifacts.read_manifest(final)
        return _report(log, StageResult("raw", manifest["id"], final, True))

    project = project_for(exp)
    manifest = artifacts.build_manifest(
        kind="raw",
        artifact="",  # filled in below: the ID is the hash of what was downloaded
        project=exp.project,
        code_version=CODE_VERSION["raw"],
        config={},
        inputs={},
    )
    with artifacts.commit_dir(final, manifest) as tmp:
        project.ingest(tmp)
        files = {
            str(p.relative_to(tmp)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in sorted(tmp.rglob("*"))
            if p.is_file()
        }
        manifest["id"] = artifacts.artifact_id(
            "raw", code_version=CODE_VERSION["raw"], config=files, inputs={}
        )
        manifest["stats"] = {"files": files}
    return _report(log, StageResult("raw", manifest["id"], final, False))


# --------------------------------------------------------------------------- prepare


def ensure_dataset(exp: Experiment, log: Log = _noop) -> StageResult:
    """Split documents by group into train/val JSONL; augment the train split only."""
    raw = ensure_raw(exp, log)
    project = project_for(exp)
    data = exp.config.data
    config = {
        "project": exp.project,
        "project_args": project.args.model_dump(mode="json"),
        "project_data_version": project.data_version,
        "val_fraction": data.val_fraction,
        "split_seed": data.split_seed,
    }
    inputs = {"raw": raw.id}
    aid = artifacts.artifact_id(
        "dataset", code_version=CODE_VERSION["dataset"], config=config, inputs=inputs
    )
    final = artifacts.slm_home() / "datasets" / exp.project / aid
    if final.is_dir():
        return _report(log, StageResult("dataset", aid, final, True))

    splits, stats = split_documents(
        project.documents(raw.path),
        data.val_fraction,
        data.split_seed,
        project.split_exclusions(),
    )
    # Augment after splitting, train only: augmenting first would put transformed copies of
    # a validation document into training, which is leakage by another name.
    train_docs = [aug for doc in splits["train"] for aug in project.augment(doc)]
    stats["train_docs_after_augment"] = len(train_docs)

    manifest = artifacts.build_manifest(
        kind="dataset",
        artifact=aid,
        project=exp.project,
        code_version=CODE_VERSION["dataset"],
        config=config,
        inputs=inputs,
        stats=stats,
    )
    with artifacts.commit_dir(final, manifest) as tmp:
        write_docs(tmp / "train.jsonl", train_docs)
        write_docs(tmp / "val.jsonl", splits["val"])
    return _report(log, StageResult("dataset", aid, final, False))


# --------------------------------------------------------------------------- tokenize


def _train_tokenizer(exp: Experiment, project: Project, dataset: Path) -> Tokenizer:
    spec = project.tokenizer_spec()
    wanted = exp.config.tokenizer.type
    if spec.type != wanted:
        raise ValueError(
            f"experiment asks for a {wanted!r} tokenizer; project supports {spec.type!r}"
        )
    if wanted == "char":
        return CharTokenizer.train(doc.text for doc in read_docs(dataset / "train.jsonl"))
    raise NotImplementedError(f"tokenizer type {wanted!r} arrives in a later milestone")


def ensure_tokenizer(exp: Experiment, log: Log = _noop) -> StageResult:
    """Fit the tokenizer on the train split only."""
    dataset = ensure_dataset(exp, log)
    config = exp.config.tokenizer.model_dump(mode="json")
    inputs = {"dataset": dataset.id}
    aid = artifacts.artifact_id(
        "tokenizer", code_version=CODE_VERSION["tokenizer"], config=config, inputs=inputs
    )
    final = artifacts.slm_home() / "tokenizers" / aid
    if final.is_dir():
        return _report(log, StageResult("tokenizer", aid, final, True))

    tokenizer = _train_tokenizer(exp, project_for(exp), dataset.path)
    manifest = artifacts.build_manifest(
        kind="tokenizer",
        artifact=aid,
        project=exp.project,
        code_version=CODE_VERSION["tokenizer"],
        config=config,
        inputs=inputs,
        stats={"vocab_size": tokenizer.vocab_size},
    )
    with artifacts.commit_dir(final, manifest) as tmp:
        tokenizer.save(tmp)
    return _report(log, StageResult("tokenizer", aid, final, False))


# --------------------------------------------------------------------------- pack


def ensure_packed(exp: Experiment, log: Log = _noop) -> StageResult:
    """Encode each split into a flat uint16 file for the trainer to memory-map."""
    tok = ensure_tokenizer(exp, log)  # also ensures (and logs) raw and dataset
    dataset = ensure_dataset(exp)
    append_eos = exp.config.data.append_eos
    config = {"append_eos": append_eos, "dtype": DTYPE.__name__}
    inputs = {"dataset": dataset.id, "tokenizer": tok.id}
    aid = artifacts.artifact_id(
        "packed", code_version=CODE_VERSION["packed"], config=config, inputs=inputs
    )
    final = artifacts.slm_home() / "packed" / aid
    if final.is_dir():
        return _report(log, StageResult("packed", aid, final, True))

    tokenizer = load_tokenizer(tok.path)
    stats: dict[str, int] = {"vocab_size": tokenizer.vocab_size}
    manifest = artifacts.build_manifest(
        kind="packed",
        artifact=aid,
        project=exp.project,
        code_version=CODE_VERSION["packed"],
        config=config,
        inputs=inputs,
        stats=stats,
    )
    with artifacts.commit_dir(final, manifest) as tmp:
        for split in SPLITS:
            split_stats = pack_split(
                read_docs(dataset.path / f"{split}.jsonl"),
                tokenizer,
                tmp / f"{split}.bin",
                append_eos=append_eos,
            )
            stats.update({f"{split}_{k}": v for k, v in split_stats.items()})
        # meta.json duplicates what the trainer needs so it never has to parse a manifest.
        (tmp / "meta.json").write_text(json.dumps({"dtype": DTYPE.__name__, **stats}, indent=2))
    return _report(log, StageResult("packed", aid, final, False))
