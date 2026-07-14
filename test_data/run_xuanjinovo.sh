#!/usr/bin/env bash

CONDA_ENV="/mnt/shared-storage-user/gaozhiqiang/miniconda3/envs/XuanjiNovo_py310"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment (must be before set -e, conda activate can return non-zero in non-interactive shells)
if [ -f "/mnt/shared-storage-user/gaozhiqiang/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/mnt/shared-storage-user/gaozhiqiang/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}" 2>/dev/null || true
fi
export PATH="${CONDA_ENV}/bin:${PATH}"

set -euo pipefail

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PEAK_PATH="${SCRIPT_DIR}/Bacillus.10k.mgf"
MODEL="${SCRIPT_DIR}/XuanjiNovo_100M_massnet.ckpt"
OUTPUT="${SCRIPT_DIR}/results"

MASTER_PORT=$(shuf -i 20000-29999 -n 1)

"${CONDA_ENV}/bin/torchrun" --nproc_per_node=1 --master_port="${MASTER_PORT}" \
    -m XuanjiNovo.XuanjiNovo \
    --mode=eval \
    --model="${MODEL}" \
    --peak_path="${PEAK_PATH}" \
    --output="${OUTPUT}"
