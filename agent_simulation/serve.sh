#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_dir/../env.sh"
model_name="${1:?Usage: serve.sh MODEL_NAME [PORT] [GPU_INDEX]}"
port="${2:-8011}"
gpu_index="${3:-1}"
case "$model_name" in
  Qwen3-32B) memory_fraction=0.80; extra=() ;;
  Qwen3.5-9B) memory_fraction=0.40; extra=(--language-model-only) ;;
  *) echo "Supported preview models: Qwen3-32B, Qwen3.5-9B" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$gpu_index"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=2
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$project_dir/models/$model_name" --served-model-name "$model_name" \
  --host 127.0.0.1 --port "$port" --gpu-memory-utilization "$memory_fraction" \
  --max-model-len 16384 --max-num-seqs 4 --max-logprobs 64 --enforce-eager \
  --no-async-scheduling --generation-config vllm \
  --attention-backend TRITON_ATTN "${extra[@]}"
