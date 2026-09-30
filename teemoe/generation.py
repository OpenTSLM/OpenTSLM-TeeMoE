"""Text generation with mixed experts through vLLM (default) or Transformers.

A request is a dict with ``prompt`` (chat-formatted), ``weights`` (one per
expert), ``samples``, ``temperature``, ``max_tokens`` and optionally ``regex``
(constrained output) and ``seed``. Both backends return, per request, the list
of generated completions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .ensemble.forecasters import child_gpus
from .model.mixture import expert_weights
from .runtime import wait_workers, worker_pool

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_vllm(requests: list[dict], *, adapters: list[Path], model: str, revision: str,
                  devices: str | None = None, python: str | None = None, max_model_len: int = 32768
                  ) -> list[list[str]]:
    """Run vLLM in worker processes (in its own environment), one per GPU in ``devices`` ("0,1")."""
    default = REPO_ROOT / ".venv-vllm/bin/python"  # created by scripts/setup.sh
    python = python or os.environ.get("TEEMOE_VLLM_PYTHON") or (str(default) if default.exists() else sys.executable)
    gpus = str(devices).split(",") if devices is not None else [None]
    shares = [requests[i::len(gpus)] for i in range(len(gpus))]  # interleaved, so similar requests spread out
    with tempfile.TemporaryDirectory(prefix="teemoe-vllm-") as work, worker_pool() as processes:
        outputs = []
        for number, (gpu, share) in enumerate(zip(gpus, shares, strict=True)):
            if not share:
                continue
            job, output = Path(work) / f"job{number}.json", Path(work) / f"output{number}.json"
            job.write_text(json.dumps(dict(model=model, revision=revision, adapters=[str(a) for a in adapters],
                                           max_model_len=max_model_len, scratch=os.environ.get("TEEMOE_LORA_SCRATCH"),
                                           requests=share)))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]))
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = child_gpus(gpu)
            processes.append(subprocess.Popen([python, "-m", "teemoe.vllm_worker", str(job), str(output)],
                                              env=env, start_new_session=True))
            outputs.append(output)
        wait_workers(processes)
        results = [None] * len(requests)
        for share, output in zip([i for i in range(len(gpus)) if shares[i]], outputs, strict=True):
            for position, row in enumerate(json.loads(output.read_text())):
                results[share + position * len(gpus)] = row["completions"]
        return results


@torch.no_grad()
def generate_transformers(model, tokenizer, requests: list[dict]) -> list[list[str]]:
    """Generate one request at a time on the in-process model with its expert mixture."""
    device = next(model.parameters()).device
    results = []
    for request in requests:
        ids = tokenizer(request["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        samples, temperature = int(request.get("samples", 1)), float(request.get("temperature", 0.0))
        if samples < 1:
            raise ValueError("samples must be positive")
        options = dict(max_new_tokens=int(request["max_tokens"]),
                       do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        if temperature > 0:
            options.update(temperature=temperature, top_k=0, top_p=1.0)
        if request.get("regex"):
            options["prefix_allowed_tokens_fn"] = _regex_constraint(tokenizer, request["regex"])
        torch.manual_seed(int(request.get("seed", 0)))
        weights = torch.tensor([request["weights"]], dtype=torch.float32, device=device)
        completions = []
        prompt_length = ids["input_ids"].shape[1]
        with expert_weights(model, weights):
            for begin in range(0, samples, 4):
                output = model.generate(**ids, **options, num_return_sequences=min(4, samples - begin))
                completions.extend(tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True))
                del output
        results.append(completions)
    return results


def _regex_constraint(tokenizer, pattern: str):
    from lmformatenforcer import RegexParser, TokenEnforcer, TokenEnforcerTokenizerData

    if not hasattr(tokenizer, "_teemoe_enforcer_data"):
        # Decode after a prefix to preserve whitespace on word-start tokens.
        prefix = tokenizer.encode("0", add_special_tokens=False)[-1]
        special = set(tokenizer.all_special_ids)
        tokens = []
        for token_id in range(len(tokenizer)):
            if token_id not in special:
                text = tokenizer.decode([prefix, token_id])[1:]
                starts_word = len(text) > len(tokenizer.decode([token_id]))
                tokens.append((token_id, text, starts_word))
        tokenizer._teemoe_enforcer_data = TokenEnforcerTokenizerData(
            tokens, lambda ids: tokenizer.decode(ids).rstrip("\ufffd"), tokenizer.eos_token_id,
            use_bitmask=False, vocab_size=len(tokenizer))
    enforcer = TokenEnforcer(tokenizer._teemoe_enforcer_data, RegexParser(pattern))
    return lambda batch_id, ids: enforcer.get_allowed_tokens(ids.tolist()).allowed_tokens
