#!/bin/bash
#SBATCH --job-name=spec_ae_train
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=50:00:00
#SBATCH --output=logs/slurm-%j.log
#SBATCH --mail-user=jsudan@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job started on $(hostname) at $(date)"

module purge
module load python/3.9.12
module load cuda/11.8.0
module load ffmpeg
source ~/myenv/bin/activate

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "----------------------------------------------------------------"

#----------------------------------------------------------------#
#  EXPERIMENT CONFIG
#----------------------------------------------------------------#
ENCODER=convnext     # cnn | convnext | vit_tiny | vit_small
MASK_TYPE=contour    # contour | energy
CONTOUR_THRESHOLD=40   # keep edges where norm magnitude > threshold/100
ENERGY_THRESHOLD=0     # keep cells within this many dB of the global max
# Set to 1 to include extra bonafide samples in training
USE_ASV5_TRAIN=0
USE_MLAAD_TRAIN=0
USE_FAMOUSFIGURES_TRAIN=1
# Set to a checkpoint path to resume training (leave empty to start fresh)
RESUME_CKPT="/scratch/hafiz_root/hafiz1/jsudan/spectrogram_ae/checkpoints/spec_ae_convnext_contour_conth_40_enth_0_asv5_0_mlaad_0_ff_1/best.pt"
EXP_NAME=spec_ae_${ENCODER}_${MASK_TYPE}_conth_${CONTOUR_THRESHOLD}_enth_${ENERGY_THRESHOLD}_asv5_${USE_ASV5_TRAIN}_mlaad_${USE_MLAAD_TRAIN}_ff_${USE_FAMOUSFIGURES_TRAIN}
EPOCHS=500
PATIENCE=10
RAWBOOST_PROB=0.7
BATCH_SIZE=64
NUM_WORKERS=8
LATENT_DIM=256

CKPT_DIR=/scratch/hafiz_root/hafiz1/jsudan/spectrogram_ae/checkpoints/${EXP_NAME}
RESULTS_DIR=/home/jsudan/spectrogram_ae/results/${EXP_NAME}

#----------------------------------------------------------------#
#  DATASET PATHS
#----------------------------------------------------------------#
ASV19_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac
ASV19_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt
ASV19_DEV_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac
ASV19_DEV_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt
ASV19_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac
ASV19_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt

ASV5_TRAIN_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_train/flac
ASV5_TRAIN_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5_train_protocol.txt
ASV5_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac
ASV5_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv

MLAAD_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/multilingual
MLAAD_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt

ITW_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild
ITW_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv

FAKEXPOSE_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/fakexpose

# FamousFigures: audio paths in the protocol are absolute, so root is empty
FAMOUSFIGURES_ROOT=
FAMOUSFIGURES_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt

#----------------------------------------------------------------#
#  TRAIN
#----------------------------------------------------------------#
mkdir -p logs "${CKPT_DIR}" "${RESULTS_DIR}"

# SLURM job names can't use shell vars in #SBATCH directives, so symlink the log
ln -sf "logs/slurm-${SLURM_JOB_ID}.log" "logs/${EXP_NAME}-${SLURM_JOB_ID}.log" 2>/dev/null || true

cd ~/spectrogram_ae

echo "EXP_NAME   : ${EXP_NAME}"
echo "CKPT_DIR   : ${CKPT_DIR}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "Training ${EXP_NAME}..."

TRAIN_ARGS=(
  --train_protocol  "${ASV19_TRAIN_PROTOCOL}"
  --train_root      "${ASV19_TRAIN_ROOT}"
  --dev_protocol    "${ASV19_DEV_PROTOCOL}"
  --dev_root        "${ASV19_DEV_ROOT}"
  --encoder         "${ENCODER}"
  --epochs          "${EPOCHS}"
  --patience        "${PATIENCE}"
  --rawboost_prob   "${RAWBOOST_PROB}"
  --batch_size      "${BATCH_SIZE}"
  --num_workers     "${NUM_WORKERS}"
  --latent_dim             "${LATENT_DIM}"
  --mask_type              "${MASK_TYPE}"
  --contour_threshold_db   "${CONTOUR_THRESHOLD}"
  --energy_threshold_db    "${ENERGY_THRESHOLD}"
  --save_dir               "${CKPT_DIR}"
)

if [ "${USE_ASV5_TRAIN}" = "1" ]; then
  TRAIN_ARGS+=(--asv5_train_root "${ASV5_TRAIN_ROOT}" --asv5_train_protocol "${ASV5_TRAIN_PROTOCOL}")
fi

if [ "${USE_MLAAD_TRAIN}" = "1" ]; then
  TRAIN_ARGS+=(--mlaad_train_root "${MLAAD_ROOT}" --mlaad_train_protocol "${MLAAD_PROTOCOL}")
fi

if [ "${USE_FAMOUSFIGURES_TRAIN}" = "1" ]; then
  TRAIN_ARGS+=(--famousfigures_train_root "${FAMOUSFIGURES_ROOT}" \
               --famousfigures_train_protocol "${FAMOUSFIGURES_PROTOCOL}")
fi

if [ -n "${RESUME_CKPT}" ]; then
  TRAIN_ARGS+=(--resume "${RESUME_CKPT}")
fi

python train.py "${TRAIN_ARGS[@]}"

CKPT="${CKPT_DIR}/best.pt"
if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] Checkpoint not found: ${CKPT}"
  exit 1
fi

echo "Training complete. Checkpoint: ${CKPT}"
echo "----------------------------------------------------------------"

#----------------------------------------------------------------#
#  EVALUATE
#----------------------------------------------------------------#
echo "Running evaluation..."

python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
  --dataset asv19 \
  --root "${ASV19_EVAL_ROOT}" \
  --protocol "${ASV19_EVAL_PROTOCOL}"

python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
  --dataset itw \
  --root "${ITW_ROOT}" \
  --protocol "${ITW_PROTOCOL}"

python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
  --dataset asv5 \
  --root "${ASV5_EVAL_ROOT}" \
  --protocol "${ASV5_EVAL_PROTOCOL}"

python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
  --dataset mlaad \
  --root "${MLAAD_ROOT}" \
  --protocol "${MLAAD_PROTOCOL}"

python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
  --dataset fakexpose \
  --root "${FAKEXPOSE_ROOT}"

python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
  --dataset famousfigures \
  --root "${FAMOUSFIGURES_ROOT}" \
  --protocol "${FAMOUSFIGURES_PROTOCOL}"

echo "----------------------------------------------------------------"
echo "Job finished at: $(date)"
