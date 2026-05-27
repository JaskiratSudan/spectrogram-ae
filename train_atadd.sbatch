#!/bin/bash
#SBATCH --job-name=spec_ae_atadd
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.log
#SBATCH --mail-user=jsudan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job started on $(hostname) at $(date)"

module purge
module load python/3.9.12
module load cuda/11.8.0
module load ffmpeg
source ~/myenv/bin/activate

echo "----------------------------------------------------------------"

cd ~/spectrogram_ae

#----------------------------------------------------------------#
#  EXPERIMENT SETTINGS — edit here
#----------------------------------------------------------------#
ENCODER=cnn               # cnn | vit_tiny
MASK_TYPE=contour         # contour | energy
CONTOUR_THRESHOLD=40
ENERGY_THRESHOLD=40

EXP_NAME=spec_ae_atadd_${ENCODER}_${MASK_TYPE}

BASE=/scratch/hafiz_root/hafiz1/jsudan/spectrogram_ae
CKPT_DIR=${BASE}/checkpoints/${EXP_NAME}
RESULTS_DIR=/home/jsudan/spectrogram_ae/results/${EXP_NAME}

#----------------------------------------------------------------#
#  DATASET PATHS
#----------------------------------------------------------------#
ATADD_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AT-ADD-Track1
ATADD_TRAIN_ROOT=${ATADD_ROOT}/train
ATADD_TRAIN_PROTOCOL=${ATADD_ROOT}/label/train.csv
ATADD_DEV_ROOT=${ATADD_ROOT}/dev
ATADD_DEV_PROTOCOL=${ATADD_ROOT}/label/dev.csv

#----------------------------------------------------------------#
#  LOG SYMLINK
#----------------------------------------------------------------#
mkdir -p logs
ln -sf "logs/slurm-${SLURM_JOB_ID}.log" "logs/${EXP_NAME}-${SLURM_JOB_ID}.log"

echo "EXP_NAME  : ${EXP_NAME}"
echo "ENCODER   : ${ENCODER}"
echo "MASK_TYPE : ${MASK_TYPE}"
echo "CKPT_DIR  : ${CKPT_DIR}"
echo "RESULTS   : ${RESULTS_DIR}"
echo "================================================================"

#----------------------------------------------------------------#
#  TRAIN
#----------------------------------------------------------------#
mkdir -p "${CKPT_DIR}"

TRAIN_ARGS=(
    --train_root        "${ATADD_TRAIN_ROOT}"
    --train_protocol    "${ATADD_TRAIN_PROTOCOL}"
    --train_dataset     atadd
    --val_frac          0.2
    --encoder           "${ENCODER}"
    --mask_type         "${MASK_TYPE}"
    --contour_threshold_db "${CONTOUR_THRESHOLD}"
    --energy_threshold_db  "${ENERGY_THRESHOLD}"
    --save_dir          "${CKPT_DIR}"
    --epochs            100
    --patience          15
    --batch_size        32
)

python -u train.py "${TRAIN_ARGS[@]}"
echo "Training complete at $(date)"
echo "================================================================"

#----------------------------------------------------------------#
#  EVALUATE on ATADD dev set
#----------------------------------------------------------------#
CKPT="${CKPT_DIR}/best.pt"
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

_eval_dataset() {
    local dataset=$1; shift
    local csv="${RESULTS_DIR}/${dataset}_scores.csv"
    if [ -f "${csv}" ] && [ -s "${csv}" ]; then
        echo "[SKIP] ${dataset}: already evaluated"
        return
    fi
    python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
        --dataset "${dataset}" "$@"
}

echo "Evaluating: ${EXP_NAME}"
echo "Checkpoint: ${CKPT}"

_eval_dataset atadd \
    --root     "${ATADD_DEV_ROOT}" \
    --protocol "${ATADD_DEV_PROTOCOL}"

echo "================================================================"
echo "All done at $(date)"
