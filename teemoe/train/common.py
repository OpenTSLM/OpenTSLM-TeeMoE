"""Small helpers shared by the trainers: distributed setup, schedule, configs, seeding."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml


@dataclass
class Distributed:
    rank: int
    world_size: int
    device: torch.device

    @property
    def main(self) -> bool:
        return self.rank == 0

    def all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            dist.all_reduce(value)
        return value

    def sync_gradients(self, parameters) -> None:
        """Sum gradients over ranks (losses are already normalized by global counts)."""
        if self.world_size == 1:
            return
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad)

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()


def setup_distributed() -> Distributed:
    """Single process, or one process per GPU under ``torchrun``."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return Distributed(rank, world_size, device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(update: int, total: int, *, peak: float, warmup_ratio: float = 0.0,
                  final_fraction: float = 1.0, schedule: str = "cosine") -> float:
    """Linear warmup, then cosine decay to ``final_fraction * peak`` (or constant)."""
    warmup = math.ceil(total * warmup_ratio)
    if warmup and update < warmup:
        return peak * (update + 1) / warmup
    if schedule == "constant" or total <= warmup:
        return peak
    progress = min(max((update - warmup + 1) / (total - warmup), 0.0), 1.0)
    return peak * (final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    return config


def log(dist_: Distributed, **values) -> None:
    if dist_.main:
        print(" ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items()),
              flush=True)
