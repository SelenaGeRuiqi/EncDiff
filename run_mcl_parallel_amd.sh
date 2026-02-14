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

# Pretrained EncDiff checkpoint (with concat, no MCL) - use your checkpoint 2
PRETRAINED_CKPT="/path/to/your/concat-encdiff/checkpoints/last.ckpt"  # TODO: fill in

# Shapes3D VQ-VAE checkpoint (referenced in config yamls, should already be correct)
# If your server paths differ from config, override via --base config or edit yamls

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"
mkdir -p logs/slurm logs/mcl_parallel

echo "[$(date)] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[$(date)] PRETRAINED_CKPT=${PRETRAINED_CKPT}"

########################################
# Parallel MCL training: 4 lambdas x 2 datasets = 8 GPUs
# GPU 0: shapes3d lambda=0.01
# GPU 1: shapes3d lambda=0.05
# GPU 2: shapes3d lambda=0.1
# GPU 3: shapes3d lambda=0.5
# GPU 4: mpi3d   lambda=0.01
# GPU 5: mpi3d   lambda=0.05
# GPU 6: mpi3d   lambda=0.1
# GPU 7: mpi3d   lambda=0.5
#
# If you only want shapes3d, comment out the mpi3d block
# and change GPU assignments accordingly.
########################################

LAMBDAS=(0.01 0.05 0.1 0.5)
LAMBDA_STRS=(001 005 010 050)

# ----- Shapes3D (GPUs 0-3) -----
SHAPES_CKPT="${PRETRAINED_CKPT}"  # concat EncDiff checkpoint for shapes3d
# If you have separate checkpoints per dataset, set them here:
# MPI3D_CKPT="/path/to/mpi3d-concat-encdiff/checkpoints/last.ckpt"
MPI3D_CKPT="${PRETRAINED_CKPT}"

echo ""
echo "================================================================"
echo " Launching 8 parallel MCL training runs"
echo " Shapes3D: GPUs 0-3, lambdas: ${LAMBDAS[*]}"
echo " MPI3D:    GPUs 4-7, lambdas: ${LAMBDAS[*]}"
echo "================================================================"

for i in 0 1 2 3; do
    lambda=${LAMBDAS[$i]}
    lambda_str=${LAMBDA_STRS[$i]}
    gpu_id=$i
    config="${CONFIG_DIR}/shapes3d-vq-4-16-encdiff-mcl-lambda${lambda_str}.yaml"
    exp_name="shapes3d-mcl-lambda${lambda_str}"
    log_file="logs/mcl_parallel/${exp_name}.log"

    echo "[$(date)] GPU ${gpu_id}: shapes3d lambda=${lambda} -> ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python main_val.py \
        -b "${config}" \
        -t \
        --gpus 0, \
        -r "${SHAPES_CKPT}" \
        --name "${exp_name}" \
        > "${log_file}" 2>&1 &
done

# ----- MPI3D (GPUs 4-7) -----
for i in 0 1 2 3; do
    lambda=${LAMBDAS[$i]}
    lambda_str=${LAMBDA_STRS[$i]}
    gpu_id=$((i + 4))
    config="${CONFIG_DIR}/mpi3d-vq-4-16-encdiff-mcl-lambda${lambda_str}.yaml"
    exp_name="mpi3d-mcl-lambda${lambda_str}"
    log_file="logs/mcl_parallel/${exp_name}.log"

    echo "[$(date)] GPU ${gpu_id}: mpi3d lambda=${lambda} -> ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python main_val.py \
        -b "${config}" \
        -t \
        --gpus 0, \
        -r "${MPI3D_CKPT}" \
        --name "${exp_name}" \
        > "${log_file}" 2>&1 &
done

echo ""
echo "[$(date)] All 8 training runs launched. Waiting for completion..."
wait

echo ""
echo "[$(date)] All runs finished."
echo "Check logs in: logs/mcl_parallel/"
echo "Check WandB:   https://wandb.ai/sege-uc-san-diego/EncDiff-MCL"
