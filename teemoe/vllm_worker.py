"""vLLM text generation with request-specific expert mixtures.

Run in the vLLM environment (vLLM pins its own transformers version):

    python -m teemoe.vllm_worker job.json output.json

Each request's mixture is baked into one exact LoRA (expert ranks concatenated)
and served through vLLM's ordinary LoRA path.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from .runtime import adapter_scratch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# vLLM compiles some kernels on first use with tools installed next to this interpreter (ninja).
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def _register_split_qkv_lora() -> None:
    """Let vLLM's fused recurrent projection accept LoRA weights for its q/k/v parts."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    for cls in (Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration):
        mapping = dict(cls.packed_modules_mapping)
        mapping["in_proj_qkvz"] = ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"]
        cls.packed_modules_mapping = mapping


class SplitQKVMapping:  # vLLM worker extension: runs the registration inside every worker
    _register_split_qkv_lora()


def run(job: dict) -> list[dict]:
    from huggingface_hub import hf_hub_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    from teemoe.model.mixture import mixture_tensors, read_lora, save_lora, vllm_names

    _register_split_qkv_lora()
    config = json.loads(Path(hf_hub_download(job["model"], "config.json", revision=job["revision"])).read_text())
    text_config = config.get("text_config", config)
    engine = LLM(model=job["model"], revision=job["revision"], tokenizer_revision=job["revision"],
                 runner="generate", dtype="bfloat16", tensor_parallel_size=job.get("tensor_parallel_size", 1),
                 gpu_memory_utilization=job.get("gpu_memory_utilization", 0.85),
                 max_model_len=job.get("max_model_len", 32768), max_num_seqs=32, seed=job.get("seed", 0),
                 enable_lora=True, max_lora_rank=64, max_loras=1, max_cpu_loras=1, lora_dtype="bfloat16",
                 worker_extension_cls="teemoe.vllm_worker.SplitQKVMapping")

    def params(request):
        kwargs = dict(n=int(request.get("samples", 1)), temperature=float(request.get("temperature", 0.0)),
                      max_tokens=int(request["max_tokens"]), seed=int(request.get("seed", 0)))
        if request.get("regex"):
            kwargs["structured_outputs"] = StructuredOutputsParams(regex=request["regex"])
        return SamplingParams(**kwargs)

    results: dict[int, list[str]] = {}
    groups: dict[tuple, list[int]] = {}
    for index, request in enumerate(job["requests"]):
        groups.setdefault(tuple(round(float(w), 12) for w in request["weights"]), []).append(index)
    experts = [read_lora(path) for path in job["adapters"]]  # loaded once
    with ExitStack() as storage:
        scratch = None
        for number, (weights, indices) in enumerate(groups.items(), start=1):
            tensors, config = mixture_tensors(experts, list(weights))
            tensors = vllm_names(tensors, text_config)
            if scratch is None:
                size = sum(t.numel() * t.element_size() for t in tensors.values())
                scratch = storage.enter_context(adapter_scratch(size, job.get("scratch")))
            adapter = save_lora(tensors, config, Path(scratch) / f"mixture{number}")
            outputs = engine.generate([job["requests"][i]["prompt"] for i in indices],
                                      [params(job["requests"][i]) for i in indices], use_tqdm=False,
                                      lora_request=LoRARequest(f"mixture{number}", number, str(adapter)))
            for i, output in zip(indices, outputs, strict=True):
                results[i] = [completion.text for completion in output.outputs]
            engine.llm_engine.remove_lora(number)
            for file in adapter.iterdir():
                file.unlink()
            if number % 20 == 0 or number == len(groups):
                print(json.dumps(dict(completed=len(results), total=len(job["requests"]))), flush=True)
    return [dict(completions=results[i]) for i in range(len(job["requests"]))]


def main() -> None:
    job_path, output_path = map(Path, sys.argv[1:3])
    results = run(json.loads(job_path.read_text()))
    output_path.write_text(json.dumps(results) + "\n")


if __name__ == "__main__":
    main()
