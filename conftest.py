"""Shared pytest fixtures for engine tests (tests/) and project tests (projects/*/tests/)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent


@pytest.fixture
def slm_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty $SLM_HOME per test, so tests never touch real artifacts."""
    home = tmp_path / "slm"
    monkeypatch.setenv("SLM_HOME", str(home))
    return home


# A minimal project that needs no network: 20 "tunes", 3 documents each (settings of the same
# tune share a group), and an augmentation that is easy to spot (upper-casing).
TOY_PROJECT = textwrap.dedent(
    """
    from pydantic import BaseModel, ConfigDict

    from slmkit.project_api import Doc, EvalPrompt, Project, SFTExample, TokenizerSpec
    from slmkit.registry import register_project


    class ToyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        augment: bool = False


    @register_project("toy")
    class Toy(Project):
        Args = ToyArgs

        def ingest(self, raw_dir):
            lines = [f"tune{t}|setting{s}|abc {t} {s} xyz" for t in range(20) for s in range(3)]
            (raw_dir / "tunes.txt").write_text("\\n".join(lines))

        def documents(self, raw_dir):
            for line in (raw_dir / "tunes.txt").read_text().splitlines():
                tune, setting, text = line.split("|")
                yield Doc(id=f"{tune}/{setting}", group=tune, text=text)

        def augment(self, doc):
            yield doc
            if self.args.augment:
                yield Doc(id=doc.id + "/up", group=doc.group, text=doc.text.upper())

        def split_exclusions(self):
            return {"tune19"}

        def tokenizer_spec(self):
            return TokenizerSpec(type="char")

        def eval_prompts(self, split):
            yield EvalPrompt(id="p", prompt="abc")

        def graders(self):
            return []

        def sft_examples(self, docs):
            # The "request" is the first three characters; the answer is the rest.
            for d in docs:
                yield SFTExample(prompt=d.text[:3], completion=d.text[3:])

        def sft_eval_prompts(self, split):
            yield EvalPrompt(id="p", prompt="abc")
    """
)

TOY_EXPERIMENT = textwrap.dedent(
    """
    run: {name: toy-base}
    project: {name: toy, args: {}}
    tokenizer: {type: char}
    data: {block_size: 8, val_fraction: 0.2}
    model: {preset: nano}
    train: {preset: default, max_tokens: 10000, batch_size: 4, lr: 1.0e-3}
    """
)


@pytest.fixture
def toy_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slm_home: Path) -> Path:
    """A repo layout with the real presets and one synthetic project, `toy/base`."""
    repo = tmp_path / "repo"
    (repo / "projects" / "toy" / "experiments").mkdir(parents=True)
    (repo / "presets").symlink_to(REPO / "presets")
    (repo / "projects" / "toy" / "project.py").write_text(TOY_PROJECT)
    (repo / "projects" / "toy" / "experiments" / "base.yaml").write_text(TOY_EXPERIMENT)
    monkeypatch.setenv("SLM_REPO", str(repo))
    return repo


@pytest.fixture
def toy_run(toy_repo: Path) -> str:
    """A briefly trained `toy/base` run (CPU, a few seconds). Returns its run ID."""
    from slmkit.config import load_experiment
    from slmkit.train.trainer import Trainer

    exp = load_experiment(
        "toy/base",
        ["run.tracker=none", "train.compile=false", "train.max_tokens=640",
         "train.warmup_tokens=64", "train.eval_every_steps=10", "train.eval_iters=2",
         "train.eval_samples=1", "train.sample_tokens=4"],
    )  # fmt: skip
    trainer = Trainer(exp, device="cpu", log=lambda _: None)
    trainer.run()
    return trainer.run_id
