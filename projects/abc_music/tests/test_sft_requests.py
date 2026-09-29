"""abc_music SFT: requests are well-formed and use only characters the model knows."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from slmkit.project_api import Doc
from slmkit.registry import load_project

PROJECT = load_project("abc_music", {}, Path("/tmp"))
proj = importlib.import_module("slmkit_projects.abc_music.project")
abcn = importlib.import_module("slmkit_projects.abc_music.abcnotation")


def test_requests_read_naturally() -> None:
    g = abcn.parse_key("G")
    assert proj.request("jig", "6/8", g) == "% a jig in G major\n"
    assert proj.request("air", "3/4", abcn.parse_key("Em")) == "% an air in E minor\n"
    assert proj.request(None, "3/4", abcn.parse_key("Ador")) == "% a tune in 3/4 time in A dorian\n"


def test_sft_requests_use_only_known_characters() -> None:
    """The char vocabulary is fixed at pretraining. A request character outside it becomes
    <unk>, which the model never learned. Every template x every rhythm and key in the corpus
    must stay inside the characters the corpus itself contains."""
    music21 = pytest.importorskip("music21")
    tunes = proj.load_tunes(Path(music21.__file__).parent / "corpus")
    known = {c for t in tunes for c in t.render()}
    for t in tunes:
        key = abcn.parse_key(t.key)
        if key is None:
            continue
        n = len(proj.WITH_RHYTHM if t.rhythm else proj.WITHOUT_RHYTHM)
        for i in range(n):
            text = proj.request(t.rhythm, t.meter, key, (i + 0.5) / n)
            assert set(text) <= known, (text, set(text) - known)


def test_examples_answer_with_every_header() -> None:
    doc = Doc(id="x#1", group="g", text="M:6/8\nL:1/8\nABc|\n",  # header-dropped render
              meta={"rhythm": "jig", "meter": "6/8", "unit": "1/8", "key": "G",
                    "body": "|GAB AGE|FGA d2B|", "changes_mid_tune": False})  # fmt: skip
    [ex] = PROJECT.sft_examples([doc])
    assert ex.prompt.startswith("% ") and "jig" in ex.prompt and "G major" in ex.prompt
    assert ex.completion.startswith("R:jig\nM:6/8\nL:1/8\nK:G\n")  # headers restored
    assert ex.meta == {"rhythm": "jig", "meter": "6/8", "key": "G"}


def test_sft_eval_prompts_mirror_the_header_prompts() -> None:
    words = list(PROJECT.sft_eval_prompts("val"))
    headers = list(PROJECT.eval_prompts("val"))
    assert [p.id for p in words] == [p.id for p in headers]
    assert [p.meta for p in words] == [p.meta for p in headers]


def test_sft_eval_prompts_state_what_the_graders_check() -> None:
    """The bars grader checks the meter and the tonic grader the key, so every eval request
    must say both; a model can't be marked down for a requirement nobody stated."""
    for p in PROJECT.sft_eval_prompts("val"):
        assert f"{p.meta['meter']} time" in p.prompt, p.prompt
        assert p.meta["key"][0] in p.prompt, p.prompt
