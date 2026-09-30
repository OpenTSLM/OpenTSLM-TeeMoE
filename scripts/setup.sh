#!/usr/bin/env bash
# Create the Python environments and check out the pinned third-party code.
#
#   bash scripts/setup.sh            # everything
#   bash scripts/setup.sh inference  # only what TeeMoE inference needs
#
# Environments (Python 3.12, uv):
#   .venv         TeeMoE itself: Qwen backbone, training, data, evaluation
#   .venv-vllm    vLLM text generation (vLLM pins its own torch/transformers)
#   .venv-fm      the forecasting models; incompatible libraries use their own:
#   .venv-moirai
#   .venv-timer
#   .venv-tirex
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"

checkout() {  # repository, revision, directory
  if [ ! -d "$3/.git" ]; then git clone --quiet "$1" "$3"; fi
  git -C "$3" fetch --quiet origin "$2" 2>/dev/null || true
  git -C "$3" checkout --quiet --detach "$2"
}
environment() {  # directory, packages or requirement files...
  local dir="$1"; shift
  echo "Setting up $dir"
  [ -x "$dir/bin/python" ] || uv venv --quiet --python 3.12 "$dir"
  uv pip install --quiet --python "$dir/bin/python" "$@"
  uv pip check --python "$dir/bin/python"
}
worker_environment() {  # directory, requirements file
  local dir="$1" requirements="$2"
  echo "Setting up $dir"
  [ -x "$dir/bin/python" ] || uv venv --quiet --python 3.12 "$dir"
  uv pip compile --quiet --no-config --python "$dir/bin/python" "$requirements" \
    --output-file "$dir/requirements.txt"
  uv pip sync --quiet --no-config --python "$dir/bin/python" "$dir/requirements.txt"
  uv pip check --no-config --python "$dir/bin/python"
}

environment .venv -e ".[kernels]"
worker_environment .venv-vllm requirements/vllm.txt
worker_environment .venv-fm requirements/forecasters.txt
worker_environment .venv-moirai requirements/moirai.txt
worker_environment .venv-timer requirements/timer.txt
worker_environment .venv-tirex requirements/tirex.txt
checkout https://github.com/ibm-granite/granite-tsfm.git 31c7a6bd640c41e53777c0afe3a13d24e00f7fe7 third_party/granite-tsfm
[ "$what" = inference ] && exit 0

environment .venv -e ".[kernels,eval]"
checkout https://github.com/SalesforceAIResearch/gift-eval.git 4d5ab3fa0fe7451bbf59bb1ff6dd76e6e414d64a third_party/gift-eval
checkout https://github.com/ServiceNow/context-is-key-forecasting.git 73f46016f8c5643bf6799ca31875eab3e8d0d075 third_party/context-is-key
checkout https://github.com/NetManAIOps/ChatTS.git a16ca1a7bd2d0cbe1dd40af37cb5658e7008357e third_party/ChatTS
echo "Environments ready. Download the GIFT-Eval data with:"
echo "  .venv/bin/hf download Salesforce/GiftEval --repo-type dataset --local-dir data/gift-eval"
