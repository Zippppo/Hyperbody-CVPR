#!/bin/bash
#SBATCH --job-name=s2i_graph_dist
#SBATCH --partition=short
#SBATCH --nodelist=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=slurm/logs/slurm_%j.log
#SBATCH --error=slurm/logs/slurm_%j.err

set -eo pipefail

source /home/comp/csrkzhu/miniconda3/etc/profile.d/conda.sh
conda activate pasco

cd /home/comp/csrkzhu/code/HyperBody-precompute-graph
mkdir -p slurm/logs

DATASET_DIR="S2I_Dataset"
DATA_DIR="${DATASET_DIR}/train"
SPLIT_FILE="${DATASET_DIR}/dataset_split_graph_precompute.json"

python - "${DATA_DIR}" "${SPLIT_FILE}" <<'PY'
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
split_file = Path(sys.argv[2])

filenames = sorted(path.name for path in data_dir.glob("*.npz"))
if not filenames:
    raise RuntimeError(f"No .npz files found in {data_dir}")

split_file.write_text(
    json.dumps({"train": filenames, "val": [], "test": []}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Wrote {len(filenames)} training files to {split_file}")
PY

python scripts/precompute_graph_distance.py \
    --tree-file "${DATASET_DIR}/tree.json" \
    --dataset-info "${DATASET_DIR}/dataset_info.json" \
    --data-dir "${DATA_DIR}" \
    --split-file "${SPLIT_FILE}" \
    --output-dir "${DATASET_DIR}" \
    --volume-size 144 128 268 \
    --dilation-radius 3 \
    --lambda 0.4 \
    --epsilon 0.01 \
    --class-batch-size 0 \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --device cuda
