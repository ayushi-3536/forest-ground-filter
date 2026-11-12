#!/bin/bash
# Example: ground masks for a large folder of chunks, one shard per array task.
#   IN=/path/chunks OUT=/path/ground_masks sbatch --array=0-7 examples/slurm_array.sbatch
#SBATCH --job-name=ground_filter
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
forest-ground-filter --in "$IN" --out "$OUT" --pattern "${PATTERN:-*.laz}" --mask-only \
  --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT" --workers "$SLURM_CPUS_PER_TASK"
