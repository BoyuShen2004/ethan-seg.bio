#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shenb@bc.edu


set -euo pipefail
echo "=== NODE: $(hostname) | START: $(date) ==="

source /projects/weilab/shenb/miniconda3/etc/profile.d/conda.sh
conda activate /projects/weilab/seg.bio/ethan/envs/nnunet-seg.bio
echo "[ENV] Python: $(which python)"; python -V


export nnUNet_raw="/projects/weilab/seg.bio/ethan/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/projects/weilab/seg.bio/ethan/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/projects/weilab/seg.bio/ethan/nnUNet/nnUNet_results"
export nnUNet_n_proc_DA=12   # good default for 1 GPU + 8 CPU cores


# Variables provided via sbatch --export
: "${DATASET_ID:?Must set DATASET_ID (e.g. 001)}"
: "${CONFIG:?Must set CONFIG (e.g. 2d, 3d_fullres, 3d_lowres)}"
: "${MODALITY:?Must set MODALITY (e.g. MRI, CT)}"
: "${TARGET:?Must set TARGET (e.g. tumor, organ)}"
: "${IN_DIR:?Must set IN_DIR}"
: "${OUT_DIR:?Must set OUT_DIR}"


echo "DATASET_ID=$DATASET_ID CONFIG=$CONFIG MODALITY=$MODALITY TARGET=$TARGET"
echo "IN_DIR=$IN_DIR"
echo "OUT_DIR=$OUT_DIR"
nvidia-smi || true


echo ">>> Training: D=$DATASET_ID | C=$CONFIG"
srun nnUNetv2_predict \
    -i "$IN_DIR" \
    -o "$OUT_DIR" \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -f all \
    --save_probabilities

echo "=== END: $(date) ==="