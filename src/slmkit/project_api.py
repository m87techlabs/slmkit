"""The Project API: the one extension point between the engine and a use case.

A project (one LLM, e.g. `projects/abc_music/`) subclasses `Project` and implements the
abstract methods. The engine only ever talks to this interface, which is what lets a new use
case be a new directory with no engine changes. In Terraform terms, this is the provider
schema and a project is a root module that satisfies it.

This is the M1 version: enough for one project. M3 (chess) is expected to stress it; any
extension must stay generic and gets an ADR (CONTRIBUTING.md rule 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class Doc:
    """One training document.

    `group` is the leakage-free split key: every Doc sharing a group lands in the same split.
    For ABC it is the tune ID (so two settings of one tune never straddle train and val); for
    chess the game ID. Getting this wrong is how validation loss ends up measuring memory
    instead of generalisation.
    """

    id: str
    group: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SFTExample:
    """One supervised example: a request and the answer the model should give to it."""

    prompt: str
    completion: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalPrompt:
    """A prompt the model is asked to continue at eval time, plus anything graders need."""

    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizerSpec:
    """What kind of tokenizer a project needs.

    `char` or `bpe`: the project's documents are text, and each experiment chooses either
    (`tokenizer.type`), because that choice is a hyperparameter worth comparing. `fixed`: the
    project defines the complete vocabulary itself (`tokens`), e.g. every legal chess move.
    """

    type: Literal["char", "bpe", "fixed"]
    vocab_size: int | None = None  # bpe
    tokens: tuple[str, ...] | None = None  # fixed: the complete vocabulary, in ID order


# A grader is a pure function: (prompt, generated text) -> named scores.
Grader = Callable[[EvalPrompt, str], dict[str, float]]

# Adjusts next-token logits during decoding (e.g. mask illegal chess moves). Typed loosely here
# so the engine's API does not import torch; the sampler defines the concrete signature (M1C).
LogitsProcessor = Callable[..., Any]


class Project(ABC):
    """Base class for every project. Register subclasses with `@register_project(name)`."""

    name: ClassVar[str]
    Args: ClassVar[type[BaseModel]] = BaseModel  # project-specific args from the YAML
    # Bump when a change to `documents()` or `augment()` would change their output. It is
    # hashed into the dataset artifact ID, so stale datasets are rebuilt rather than reused.
    data_version: ClassVar[int] = 1

    def __init__(self, args: BaseModel, home: Path) -> None:
        self.args = args
        self.home = home  # $SLM_HOME

    @abstractmethod
    def ingest(self, raw_dir: Path) -> None:
        """Fetch source data into `raw_dir`. Called once; `raw_dir` is then immutable."""

    @abstractmethod
    def documents(self, raw_dir: Path) -> Iterator[Doc]:
        """Yield every document, each with a meaningful `group`."""

    def augment(self, doc: Doc) -> Iterator[Doc]:
        """Expand one training document. Called on the train split only, after splitting."""
        yield doc

    def split_exclusions(self) -> set[str]:
        """Doc IDs or groups to drop entirely, e.g. games later used as eval puzzles."""
        return set()

    @abstractmethod
    def tokenizer_spec(self) -> TokenizerSpec: ...

    def sft_examples(self, docs: Iterable[Doc]) -> Iterator[SFTExample] | None:
        """Instruction/answer pairs built from documents (one split at a time), or None if the
        project has no SFT stage. The loss is computed on `completion` only."""
        return None

    def sft_eval_prompts(self, split: str) -> Iterator[EvalPrompt] | None:
        """Requests in the same form as `sft_examples` prompts, for sampling and grading a
        fine-tuned model. `meta` should say what was asked (as in `eval_prompts`), so the same
        graders measure whether the model did it. None if the project has no SFT stage."""
        return None

    @abstractmethod
    def eval_prompts(self, split: str) -> Iterator[EvalPrompt]: ...

    @abstractmethod
    def graders(self) -> list[Grader]:
        """Domain graders only; generic ones come from `slmkit.graders`."""

    def logits_processor(self) -> LogitsProcessor | None:
        return None

    def render_sample(self, prompt: EvalPrompt, text: str) -> dict[str, bytes]:
        """Files that show one generated sample to a person, keyed by file extension, for
        `slm export --to-windows` (ADR 0007). `text` is the prompt plus the model's output.
        Default: the text itself. A project can add a rendering a person can use directly,
        e.g. MIDI for music or a PGN for chess."""
        return {".txt": text.encode()}

    def web_viewer(self) -> Path | None:
        """A directory of static files that draws this project's outputs in `slm serve`'s web
        page, or None. It must contain `viewer.js`, an ES module (see
        `src/slmkit/serve/static/index.html` for the functions it may export). `slm export`
        copies it into the model as `ui/`, so the model carries its own viewer and the server
        never loads project code (ADR 0008)."""
        return None
