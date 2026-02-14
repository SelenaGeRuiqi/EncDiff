#!/bin/bash
#SBATCH --job-name=encdiff-mcl
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --export=ALL

echo "[$(date)] Running on host: $(hostname)"
echo "[$(date)] SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

########################################
# Environment setup
########################################
source ~/.bashrc
source ~/slurm_tools/mi.sh
conda activate encdiff-new
mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER

########################################
# Project paths - EDIT THESE
########################################
PROJECT_ROOT="/vast/users/guangyi.chen/causal_group/selena/EncDiff"
CONFIG_DIR="${PROJECT_ROOT}/configs/mcl"

# Pretrained EncDiff checkpoint (with concat, no MCL) for MPI3D
MPI3D_CKPT="/path/to/your/mpi3d-concat-encdiff/checkpoints/last.ckpt"  # TODO: fill in

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm logs/mcl_parallel

echo "[$(date)] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[$(date)] MPI3D_CKPT=${MPI3D_CKPT}"

########################################
# MPI3D: 5 loss types x 2 lambdas = 10 runs
# Round 1: 8 GPUs (5 x lambda=0.01 + 3 x lambda=0.05)
# Round 2: 2 GPUs (remaining 2 x lambda=0.05)
#
# Loss type short names:
#   nce      = nce_logistic
#   infonce  = infonce_mechgrad
#   fisher   = fisher_sm
#   denoise  = denoise_sm
#   jacobian = jacobian_vjp_infonce
########################################

echo ""
echo "================================================================"
echo " Round 1: 8 parallel MCL runs on MPI3D"
echo " GPU 0-4: all 5 losses x lambda=0.01"
echo " GPU 5-7: nce/infonce/fisher x lambda=0.05"
echo "================================================================"

# --- Round 1: lambda=0.01 for all 5 losses (GPU 0-4) ---
LOSS_SHORTS_ALL=(nce infonce fisher denoise jacobian)
gpu_id=0
for loss_short in "${LOSS_SHORTS_ALL[@]}"; do
    config="${CONFIG_DIR}/mpi3d-mcl-${loss_short}-lambda001.yaml"
    exp_name="mpi3d-${loss_short}-lambda001"
    log_file="logs/mcl_parallel/${exp_name}.log"

    echo "[$(date)] GPU ${gpu_id}: ${loss_short} lambda=0.01 -> ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python main_val.py \
        -b "${config}" \
        -t \
        --gpus 0, \
        -r "${MPI3D_CKPT}" \
        --name "${exp_name}" \
        > "${log_file}" 2>&1 &

    gpu_id=$((gpu_id + 1))
done

# --- Round 1: lambda=0.05 for first 3 losses (GPU 5-7) ---
LOSS_SHORTS_R1=(nce infonce fisher)
for loss_short in "${LOSS_SHORTS_R1[@]}"; do
    config="${CONFIG_DIR}/mpi3d-mcl-${loss_short}-lambda005.yaml"
    exp_name="mpi3d-${loss_short}-lambda005"
    log_file="logs/mcl_parallel/${exp_name}.log"

    echo "[$(date)] GPU ${gpu_id}: ${loss_short} lambda=0.05 -> ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python main_val.py \
        -b "${config}" \
        -t \
        --gpus 0, \
        -r "${MPI3D_CKPT}" \
        --name "${exp_name}" \
        > "${log_file}" 2>&1 &

    gpu_id=$((gpu_id + 1))
done

echo ""
echo "[$(date)] Round 1: 8 runs launched. Waiting..."
wait

echo ""
echo "================================================================"
echo " Round 2: 2 remaining runs (denoise/jacobian x lambda=0.05)"
echo "================================================================"

# --- Round 2: lambda=0.05 for remaining 2 losses (GPU 0-1) ---
LOSS_SHORTS_R2=(denoise jacobian)
gpu_id=0
for loss_short in "${LOSS_SHORTS_R2[@]}"; do
    config="${CONFIG_DIR}/mpi3d-mcl-${loss_short}-lambda005.yaml"
    exp_name="mpi3d-${loss_short}-lambda005"
    log_file="logs/mcl_parallel/${exp_name}.log"

    echo "[$(date)] GPU ${gpu_id}: ${loss_short} lambda=0.05 -> ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python main_val.py \
        -b "${config}" \
        -t \
        --gpus 0, \
        -r "${MPI3D_CKPT}" \
        --name "${exp_name}" \
        > "${log_file}" 2>&1 &

    gpu_id=$((gpu_id + 1))
done

echo ""
echo "[$(date)] Round 2: 2 runs launched. Waiting..."
wait

echo ""
echo "[$(date)] All 10 runs finished."
echo "Check logs in: logs/mcl_parallel/"
echo "Check WandB:   https://wandb.ai/sege-uc-san-diego/EncDiff-MCL"
