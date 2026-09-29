"""`slm serve`: an exported model behind a small HTTP API.

    GET  /health     liveness: {"status": "ok", "model": "abc-folk:1"}
    GET  /info       what the model is, how to prompt it, the default sampling settings
    POST /generate   {"prompt": "...", "max_new_tokens"?, "temperature"?, "top_k"?, "top_p"?,
                      "seed"?} -> the completion, why it stopped, and timings

It serves an *export* (`models/<name>/<version>/`), never a run directory: what is served is
exactly what was checked for parity and described on the model card, and it can't change
while the server is up.

Deliberately small. One model per process, one request at a time (a lock around generation:
the models are tiny, and a queue in front of one CPU core buys nothing), no auth, no TLS, bound
to 127.0.0.1 by default. For anything reachable by others, put an existing gateway in front of
it (DESIGN §6.9).
"""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from slmkit.export.hf import ExportedModel
from slmkit.sampling import generate
from slmkit.tokenizers import EOS_ID, UNK_ID

MAX_NEW_TOKENS = 4096  # a hard ceiling per request, so one call can't hold the server for long


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt field is an error, not ignored

    prompt: str = Field("", max_length=20_000)
    max_new_tokens: int | None = Field(None, ge=1, le=MAX_NEW_TOKENS)
    temperature: float | None = Field(None, ge=0.0, le=5.0)  # 0 = greedy
    top_k: int | None = Field(None, ge=0)  # 0 = off
    top_p: float | None = Field(None, gt=0.0, le=1.0)  # 1.0 = off
    seed: int | None = Field(None, ge=0, description="omit for a random seed (returned)")


class GenerateResponse(BaseModel):
    model: str
    completion: str  # what the model wrote, without the prompt
    text: str  # prompt + completion
    finished: bool  # True: the model emitted <eos>. False: it hit max_new_tokens
    prompt_tokens: int
    new_tokens: int
    unknown_prompt_tokens: int  # prompt text the vocabulary has no token for (became <unk>)
    seed: int
    settings: dict[str, Any]
    seconds: float
    tokens_per_second: float


def create_app(exported: ExportedModel, device: torch.device) -> FastAPI:
    model = exported.model.to(device).eval()
    tok = exported.tokenizer
    defaults = exported.generation
    stats = exported.manifest["stats"]
    lock = threading.Lock()
    app = FastAPI(
        title=f"slmkit · {exported.ref}",
        description=f"Generation API for {exported.ref}. See MODEL_CARD.md in {exported.path}.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": exported.ref}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "model": exported.ref,
            "project": exported.manifest["project"],
            "stage": stats["stage"],
            "params": stats["params"],
            "vocab_size": stats["vocab_size"],
            "context_tokens": stats["block_size"],
            "example_prompt": stats["example_prompt"],
            "defaults": defaults,
            "device": str(device),
            "source_run": exported.manifest["inputs"]["run"],
        }

    @app.post("/generate")
    def generate_(req: GenerateRequest) -> GenerateResponse:
        max_new = int(req.max_new_tokens or defaults["max_new_tokens"])
        temperature = float(defaults["temperature"] if req.temperature is None else req.temperature)
        top_k = int(defaults["top_k"] if req.top_k is None else req.top_k)
        top_p = float(defaults["top_p"] if req.top_p is None else req.top_p)
        settings = {"max_new_tokens": max_new, "temperature": temperature, "top_k": top_k,
                    "top_p": top_p}  # fmt: skip
        seed = secrets.randbelow(2**31) if req.seed is None else req.seed
        ids = tok.encode(req.prompt)
        # An empty prompt starts a fresh document: in training, every document followed <eos>.
        context = ids or [EOS_ID]
        with lock:
            start = time.perf_counter()
            gen = torch.Generator(device=device).manual_seed(seed)
            out = generate(
                model,
                torch.tensor([context], device=device),
                max_new,
                temperature=temperature,
                top_k=top_k or None,
                top_p=top_p if top_p < 1.0 else None,
                stop_token=EOS_ID,
                generator=gen,
            )[0, len(context) :].tolist()
            seconds = time.perf_counter() - start
        finished = EOS_ID in out
        new = out[: out.index(EOS_ID)] if finished else out
        completion = tok.decode(new)
        return GenerateResponse(
            model=exported.ref,
            completion=completion,
            text=req.prompt + completion,
            finished=finished,
            prompt_tokens=len(ids),
            new_tokens=len(out),
            unknown_prompt_tokens=ids.count(UNK_ID),
            seed=seed,
            settings=settings,
            seconds=round(seconds, 4),
            tokens_per_second=round(len(out) / seconds, 1) if seconds > 0 else 0.0,
        )

    return app
