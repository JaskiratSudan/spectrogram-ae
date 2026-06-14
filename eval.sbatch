#!/bin/bash
#SBATCH --job-name=spec_ae_eval
#SBATCH --partition=spgpu
#SBATCH --account=hafiz1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.log
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

BASE=/scratch/hafiz_root/hafiz1/jsudan/spectrogram_ae
RESULTS_BASE=/home/jsudan/spectrogram_ae/results

#----------------------------------------------------------------#
#  CHECKPOINTS TO EVALUATE
#  (best.pt for cnn/vit_tiny contour are corrupted — using nearest epoch ckpt)
#----------------------------------------------------------------#
declare -A CKPTS
# CKPTS[spec_ae_cnn_contour]="${BASE}/checkpoints/spec_ae_cnn/epoch_80.pt"
# CKPTS[spec_ae_vit_tiny_contour]="${BASE}/checkpoints/spec_ae_vit_tiny/epoch_85.pt"
# CKPTS[spec_ae_cnn_energy]="${BASE}/checkpoints/spec_ae_cnn_energy_asv5_0_mlaad_0/best.pt"
CKPTS[spec_ae_cnn_contour_conth_40_enth_0_asv5_0_mlaad_0]="${BASE}/checkpoints/spec_ae_cnn_contour_conth_40_enth_0_asv5_0_mlaad_0/best.pt"
CKPTS[spec_ae_cnn_contour_conth_40_enth_0_asv5_0_mlaad_0_ff_1]="${BASE}/checkpoints/spec_ae_cnn_contour_conth_40_enth_0_asv5_0_mlaad_0_ff_1/best.pt"


#----------------------------------------------------------------#
#  DATASET PATHS
#----------------------------------------------------------------#
ASV19_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac
ASV19_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt

ASV5_EVAL_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac
ASV5_EVAL_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv

MLAAD_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/multilingual
MLAAD_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt

ITW_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild
ITW_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv

FAKEXPOSE_ROOT=/nfs/turbo/umd-hafiz/issf_server_data/fakexpose

FAMOUSFIGURES_ROOT=
FAMOUSFIGURES_PROTOCOL=/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt

#  Set FORCE=1 to re-run datasets that already have a scores CSV
FORCE=1

#----------------------------------------------------------------#
#  SCORING CONFIG
#  score_type: pixel (default) | feature (CNN/ConvNeXt only)
#  feature_weights: comma-separated per-block weights (finest→coarsest)
#  normalize_features: 1 = normalize per-channel L2 before distance (recommended)
#----------------------------------------------------------------#
SCORE_TYPE=feature
FEATURE_WEIGHTS="1.0,0.8,0.4,0.2"
NORMALIZE_FEATURES=1

#----------------------------------------------------------------#
#  HELPER — skip datasets already evaluated (csv exists and non-empty)
#----------------------------------------------------------------#
_eval_dataset() {
    local dataset=$1; shift
    local csv="${RESULTS_DIR}/${dataset}_scores.csv"
    if [ "${FORCE}" != "1" ] && [ -f "${csv}" ] && [ -s "${csv}" ]; then
        echo "[SKIP] ${EXP_NAME}/${dataset}: already evaluated"
        return
    fi
    local score_args=(--score_type "${SCORE_TYPE}" --feature_weights "${FEATURE_WEIGHTS}")
    [ "${NORMALIZE_FEATURES}" != "1" ] && score_args+=(--no_normalize_features)
    python evaluate.py --checkpoint "${CKPT}" --results_dir "${RESULTS_DIR}" \
        --dataset "${dataset}" "${score_args[@]}" "$@"
}

#----------------------------------------------------------------#
#  RUN EVALUATIONS
#----------------------------------------------------------------#
for EXP_NAME in "${!CKPTS[@]}"; do
    CKPT="${CKPTS[$EXP_NAME]}"
    RESULTS_DIR="${RESULTS_BASE}/${EXP_NAME}"

    if [ ! -f "${CKPT}" ]; then
        echo "[SKIP] ${EXP_NAME}: checkpoint not found at ${CKPT}"
        continue
    fi

    mkdir -p "${RESULTS_DIR}"
    echo "================================================================"
    echo "Evaluating: ${EXP_NAME}"
    echo "Checkpoint: ${CKPT}"
    echo "Results   : ${RESULTS_DIR}"
    echo "----------------------------------------------------------------"

    _eval_dataset asv19       --root "${ASV19_EVAL_ROOT}" --protocol "${ASV19_EVAL_PROTOCOL}"
    _eval_dataset itw         --root "${ITW_ROOT}"        --protocol "${ITW_PROTOCOL}"
    _eval_dataset asv5        --root "${ASV5_EVAL_ROOT}"  --protocol "${ASV5_EVAL_PROTOCOL}"
    _eval_dataset mlaad       --root "${MLAAD_ROOT}"      --protocol "${MLAAD_PROTOCOL}"
    _eval_dataset fakexpose   --root "${FAKEXPOSE_ROOT}"
    _eval_dataset famousfigures --root "${FAMOUSFIGURES_ROOT}" --protocol "${FAMOUSFIGURES_PROTOCOL}"

    echo ""
done

echo "================================================================"
echo "All evaluations complete."
echo "Job finished at: $(date)"
