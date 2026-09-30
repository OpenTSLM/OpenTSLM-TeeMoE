#!/usr/bin/env bash
# Build the training data and train TeeMoE from scratch on one 8-GPU node.
# Every step reuses finished outputs, so the script can be rerun after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
python=.venv/bin/python
gpus=${GPUS:-8}

# 1. Training data
[ -d data/gift-eval ] || .venv/bin/hf download Salesforce/GiftEval --repo-type dataset --local-dir data/gift-eval
$python -m teemoe.data.aggregation --download            # -> data/aggregation
$python -m teemoe.data.native                            # -> data/native/train.jsonl
$python -m teemoe.data.analysis                          # -> data/analysis/train.jsonl

# 2. The three experts
$python -m teemoe.train.aggregation prepare --config configs/aggregation.yaml
[ -f artifacts/aggregation/expert/editor.safetensors ] || \
  $python -m torch.distributed.run --nproc_per_node 2 -m teemoe.train.aggregation editor --config configs/aggregation.yaml
[ -f artifacts/native/adapter/adapter_config.json ] || \
  $python -m torch.distributed.run --nproc_per_node "$gpus" -m teemoe.train.text --config configs/native.yaml --resume
[ -f artifacts/analysis/adapter/adapter_config.json ] || \
  $python -m torch.distributed.run --nproc_per_node "$gpus" -m teemoe.train.text --config configs/analysis.yaml --resume

# 3. The controller, then one loadable checkpoint
[ -f artifacts/controller/controller.safetensors ] || \
  $python -m torch.distributed.run --nproc_per_node "$gpus" -m teemoe.train.controller --config configs/controller.yaml
$python -m teemoe.checkpoint --output checkpoints/teemoe
