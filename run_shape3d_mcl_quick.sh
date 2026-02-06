#!/bin/bash
#
# MCL Exploration Training Script
# 数据集: Shapes3D (作为快速验证)
# Lambda值: 0.05, 0.1, 0.5
# GPU: 1
# 每个epoch结束自动生成swap可视化并上传到WandB
#

set -e

# ============================================================================
# 配置
# ============================================================================

PROJECT_ROOT="/mnt/data_7tb/selena/projects/EncDiff"
CONFIG_DIR="${PROJECT_ROOT}/configs/mcl"

# Shapes3D checkpoint (baseline: FactorVAE=1.0, DCI=0.992)
PRETRAINED_CKPT="/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T07-42-42_shapes3d-vq-4-16-encdiff23/checkpoints/last.ckpt"

# GPU
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Lambda values
LAMBDAS=(0.05 0.1 0.5)

# Training params
MAX_EPOCHS=3
BATCH_SIZE=128

# WandB
WANDB_PROJECT="EncDiff-MCL"
WANDB_ENTITY="sege-uc-san-diego-org"

echo "========================================================================"
echo "🚀 MCL Exploration on Shapes3D"
echo "========================================================================"
echo "⚠️  NOTE: Shapes3D baseline is already very good:"
echo "   FactorVAE: 1.0"
echo "   DCI: 0.992"
echo "   Expected improvement: minimal (this is for quick validation)"
echo ""
echo "💡 Recommendation: Use MPI3D or Cars3D for better demonstration"
echo "========================================================================"
echo ""
echo "GPU: ${GPU_ID}"
echo "Pretrained: ${PRETRAINED_CKPT}"
echo "Lambda values: ${LAMBDAS[@]}"
echo "Epochs: ${MAX_EPOCHS}"
echo "========================================================================"
echo ""

# Check checkpoint
if [ ! -f "${PRETRAINED_CKPT}" ]; then
    echo "❌ ERROR: Checkpoint not found: ${PRETRAINED_CKPT}"
    exit 1
fi

cd ${PROJECT_ROOT}

# ============================================================================
# Training Loop
# ============================================================================

START_TIME=$(date +%s)

for lambda in "${LAMBDAS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "📊 Training lambda_mcl = ${lambda}"
    echo "========================================================================"
    
    # Format lambda string (0.05 -> 005, 0.1 -> 01, 0.5 -> 05)
    lambda_str=$(echo $lambda | awk '{printf "%03d", $1*100}')
    
    CONFIG_FILE="${CONFIG_DIR}/shapes3d-vq-4-16-encdiff-mcl-lambda${lambda_str}.yaml"
    EXP_NAME="shapes3d-mcl-lambda${lambda_str}-explore"
    WANDB_RUN="shapes3d-mcl-lambda${lambda}-explore"
    
    echo "Config: ${CONFIG_FILE}"
    echo "Experiment: ${EXP_NAME}"
    echo "WandB run: ${WANDB_RUN}"
    echo ""
    
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo "❌ Config not found: ${CONFIG_FILE}"
        echo "Creating config..."
        
        # Create config on-the-fly
        mkdir -p ${CONFIG_DIR}
        cat > ${CONFIG_FILE} << EOF
model:
  base_learning_rate: 2.0e-7
  target: ldm.models.diffusion.ddpm_enc.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0155
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    loss_type: l1
    first_stage_key: "image"
    cond_stage_key: "image"
    image_size: 16
    channels: 3
    cond_stage_trainable: true
    concat_mode: False
    scale_by_std: True
    monitor: 'train/loss_simple'
    conditioning_key: crossattn
    eval_name: shapes3d
    
    lambda_mcl: ${lambda}
    mcl_tau: 0.1
    mcl_proj_dim: 128
    use_mcl: true

    scheduler_config:
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [1000]
        cycle_lengths: [10000000000000]
        f_start: [1.e-6]
        f_max: [1.]
        f_min: [1.]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel_enc.UNetModel
      params:
        image_size: 16
        in_channels: 3
        out_channels: 3
        model_channels: 64
        attention_resolutions: [1, 2, 4]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_heads: 8
        use_scale_shift_norm: True
        resblock_updown: True
        use_spatial_transformer: True
        context_dim: 16
        latent_unit: 20

    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 2048
        use_disentangled_concat: true
        disentangled_dim: 20
        monitor: "train/rec_loss"
        ckpt_path: "/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-04T07-00-13_shapes3d_vq_4_1623/checkpoints/last.ckpt"
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 64
          in_channels: 3
          out_ch: 3
          ch: 32
          ch_mult: [1, 2, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.diffusionmodules.openaimodel_enc.Encoder4
      params:
        d: 128
        context_dim: 16
        latent_unit: 20

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 128
    num_workers: 8
    wrap: True
    train:
      target: ldm.data.disdata.Shapes3DTrain
    validation:
      target: ldm.data.disdata.Shapes3DTrain

lightning:
  callbacks:
    image_logger:
      target: main_val.ImageLogger
      params:
        log_config:
          target: ldm.tools.Record
          params:
            plot_image: true
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: false
        log_images_kwargs:
          inpaint: false
          sample_swap: True
          plot_progressive_rows: False
          plot_diffusion_rows: False
    
    swap_visualization:
      target: main_val.SwapVisualizationCallback
      params:
        n_samples: 8
        ddim_steps: 200
        save_locally: true
        log_to_wandb: true

  trainer:
    benchmark: True
    max_epochs: 3
    check_val_every_n_epoch: 1
    
  logger:
    target: pytorch_lightning.loggers.WandbLogger
    params:
      name: "${WANDB_RUN}"
      project: "${WANDB_PROJECT}"
      entity: "${WANDB_ENTITY}"
      save_dir: "logs/"
      offline: False
      log_model: False
      config:
        dataset: "Shapes3D"
        lambda_mcl: ${lambda}
        baseline_factor_vae: 1.0
        baseline_dci: 0.992
EOF
        echo "✅ Config created"
    fi
    
    # Run training
    python main_val.py \
        -b ${CONFIG_FILE} \
        -t \
        --gpus ${GPU_ID}, \
        -r ${PRETRAINED_CKPT} \
        --name ${EXP_NAME} 2>&1 | tee logs/training_${EXP_NAME}.log
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "❌ Training failed for lambda=${lambda}"
        echo "Check log: logs/training_${EXP_NAME}.log"
    else
        echo ""
        echo "✅ Completed lambda=${lambda}"
    fi
    
    echo "Cooling down for 30s..."
    sleep 30
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "========================================================================"
echo "✅ Training Complete!"
echo "========================================================================"
echo "Time: ${ELAPSED} minutes"
echo ""
echo "📊 View results:"
echo "   WandB: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo ""
echo "💡 Next steps:"
echo "   1. Check WandB for swap visualizations"
echo "   2. Compare FactorVAE and DCI metrics"
echo "   3. If you want better improvements, run on MPI3D or Cars3D"
echo "========================================================================"