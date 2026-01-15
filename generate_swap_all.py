#!/usr/bin/env python
"""
为多个实验的所有 checkpoints 生成 swap 可视化图片

输入: 实验目录列表
对于每个实验目录:
  - 找到 checkpoints/ 下所有 checkpoint 文件
  - 根据实验名字判断数据集类型 (shapes3d, cars3d, mpi3d)
  - 为每个 checkpoint 生成 swap 可视化
  - 保存到 swap_images_{checkpoint名字}/ 文件夹
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import glob

sys.path.append('/mnt/data_7tb/selena/projects/EncDiff')

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.data.disdata import Shapes3DTrain, Cars3DTrain, MPI3DTrain
from torchvision.utils import make_grid


# ============================================================================
# 实验目录列表
# ============================================================================
EXPERIMENT_DIRS = [
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-04T09-59-07_shapes3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-05T01-46-21_cars3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-05T06-14-44_mpi3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T07-42-42_shapes3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T08-23-34_mpi3d-vq-4-16-encdiff23',
    '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T08-25-57_cars3d-vq-4-16-encdiff23',
]

def get_dataset_type(exp_dir):
    """根据实验目录名判断数据集类型"""
    exp_name = os.path.basename(exp_dir).lower()
    if 'shapes3d' in exp_name:
        return 'shapes3d'
    elif 'cars3d' in exp_name:
        return 'cars3d'
    elif 'mpi3d' in exp_name:
        return 'mpi3d'
    else:
        raise ValueError(f"Unknown dataset type from exp_dir: {exp_dir}")


def get_config_path_from_exp_dir(exp_dir):
    """从实验目录中获取保存的配置文件路径"""
    configs_dir = os.path.join(exp_dir, 'configs')
    if not os.path.exists(configs_dir):
        raise ValueError(f"Configs directory not found: {configs_dir}")
    
    # 查找 *-project.yaml 文件
    project_yamls = glob.glob(os.path.join(configs_dir, '*-project.yaml'))
    if not project_yamls:
        raise ValueError(f"No project.yaml found in: {configs_dir}")
    
    # 使用第一个（通常只有一个，或取最早的）
    project_yamls = sorted(project_yamls)
    return project_yamls[0]


def get_dataset(dataset_type):
    """获取数据集实例"""
    dataset_map = {
        'shapes3d': Shapes3DTrain,
        'cars3d': Cars3DTrain,
        'mpi3d': MPI3DTrain,
    }
    return dataset_map[dataset_type]()


def load_model(config_path, ckpt_path):
    """加载模型"""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    model = model.cuda()
    return model, config


def generate_swap_visualization(model, dataset, output_path, n_samples=8, ddim_steps=200):
    """
    生成 swap 可视化图片
    
    Args:
        model: 加载的模型
        dataset: 数据集
        output_path: 输出路径
        n_samples: 每次生成的样本数
        ddim_steps: DDIM 采样步数
    """
    print(f"      Generating swap visualization...")
    
    # 随机选择样本
    np.random.seed(42)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    images = []
    for idx in indices:
        sample = dataset[idx]
        if torch.is_tensor(sample['image']):
            img = sample['image']
        else:
            img = torch.from_numpy(sample['image']).float()
        images.append(img)
    
    batch_images = torch.stack(images, dim=0).cuda()  # (n_samples, H, W, C)
    batch_dict = {'image': batch_images}
    
    # 调用 log_images 生成 swap 样本
    with torch.no_grad():
        log = model.log_images(
            batch_dict,
            N=n_samples,
            sample_swap=True,
            ddim_steps=ddim_steps,
            ddim_eta=0.0,
            inpaint=False,
            plot_progressive_rows=False,
            plot_diffusion_rows=False
        )
    
    # 保存原始输入
    if 'inputs' in log:
        inputs = log['inputs']
        grid = make_grid(inputs, nrow=4)
        grid = (grid + 1) / 2  # normalize to [0,1]
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 1)
        
        input_path = output_path.replace('.png', '_inputs.png')
        Image.fromarray((grid * 255).astype(np.uint8)).save(input_path)
    
    # 保存 swap 结果
    if 'samples_swapping' in log:
        swapping = log['samples_swapping']
        latent_unit = model.model.diffusion_model.latent_unit
        
        # 保存完整的 swap grid
        grid = make_grid(swapping, nrow=n_samples)
        grid = (grid + 1) / 2
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 1)
        
        Image.fromarray((grid * 255).astype(np.uint8)).save(output_path)
        
        # 创建带标签的可视化
        create_labeled_visualization(
            log['inputs'].cpu(),
            swapping.cpu(),
            latent_unit,
            n_samples,
            output_path.replace('.png', '_labeled.png')
        )
    
    return log


def create_labeled_visualization(inputs, swapping, latent_unit, n_samples, output_path, 
                                  factors_per_page=10):
    """
    创建带标签的可视化，方便分析每个 factor
    
    图片布局：
    - 第一行: 原始输入图片
    - 后续每行: swap 第 i 个 latent unit 后的结果
    """
    
    # 分页显示
    n_pages = (latent_unit + factors_per_page - 1) // factors_per_page
    
    for page in range(n_pages):
        start_factor = page * factors_per_page
        end_factor = min((page + 1) * factors_per_page, latent_unit)
        n_factors_this_page = end_factor - start_factor
        
        fig, axes = plt.subplots(n_factors_this_page + 1, n_samples, 
                                figsize=(2*n_samples, 2*(n_factors_this_page+1)))
        
        # 第一行：原始输入
        for i in range(n_samples):
            img = inputs[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Input', fontsize=10, rotation=0, ha='right', labelpad=30)
        
        # 后续每行：swap 后的结果
        for factor_idx in range(start_factor, end_factor):
            row_idx = factor_idx - start_factor + 1
            for sample_idx in range(n_samples):
                img_idx = factor_idx * n_samples + sample_idx
                img = swapping[img_idx].permute(1, 2, 0).numpy()
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
                axes[row_idx, sample_idx].imshow(img)
                axes[row_idx, sample_idx].axis('off')
                
            axes[row_idx, 0].set_ylabel(f'Factor {factor_idx}', fontsize=10, 
                                        rotation=0, ha='right', labelpad=30)
        
        plt.tight_layout()
        page_path = output_path.replace('.png', f'_page{page+1}.png')
        plt.savefig(page_path, dpi=150, bbox_inches='tight')
        plt.close()


def process_experiment(exp_dir):
    """处理单个实验目录"""
    print(f"\n{'='*70}")
    print(f"📁 Processing: {exp_dir}")
    print(f"{'='*70}")
    
    # 判断数据集类型
    dataset_type = get_dataset_type(exp_dir)
    print(f"   Dataset type: {dataset_type}")
    
    # 从实验目录获取保存的配置文件（这样可以确保模型结构匹配）
    config_path = get_config_path_from_exp_dir(exp_dir)
    print(f"   Config: {config_path}")
    
    # 加载数据集（只加载一次）
    print(f"   Loading dataset...")
    dataset = get_dataset(dataset_type)
    print(f"   Dataset size: {len(dataset)}")
    
    # 获取所有 checkpoint 文件
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    ckpt_files = sorted(ckpt_files)
    
    print(f"   Found {len(ckpt_files)} checkpoints:")
    for ckpt_file in ckpt_files:
        print(f"      - {os.path.basename(ckpt_file)}")
    
    # 为每个 checkpoint 生成可视化
    for ckpt_path in ckpt_files:
        ckpt_name = os.path.basename(ckpt_path).replace('.ckpt', '')
        output_dir = os.path.join(exp_dir, f'swap_images_{ckpt_name}')
        
        # 检查输出目录是否已存在且非空，如果是则跳过
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"\n   ⏭️  Skipping checkpoint: {ckpt_name} (output dir not empty)")
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n   📦 Processing checkpoint: {ckpt_name}")
        print(f"      Output dir: {output_dir}")
        
        try:
            # 加载模型
            print(f"      Loading model...")
            model, config = load_model(config_path, ckpt_path)
            
            # 生成可视化
            output_path = os.path.join(output_dir, 'swap_visualization.png')
            generate_swap_visualization(
                model,
                dataset,
                output_path,
                n_samples=8,
                ddim_steps=200
            )
            
            print(f"      ✅ Done: {output_dir}")
            
            # 释放 GPU 内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    print("=" * 70)
    print("🎨 Generate Swap Visualization for All Checkpoints")
    print("=" * 70)
    
    print(f"\nExperiment directories to process: {len(EXPERIMENT_DIRS)}")
    for exp_dir in EXPERIMENT_DIRS:
        print(f"   - {exp_dir}")
    
    # 处理每个实验
    for exp_dir in EXPERIMENT_DIRS:
        if os.path.exists(exp_dir):
            process_experiment(exp_dir)
        else:
            print(f"\n⚠️ Directory not found: {exp_dir}")
    
    print("\n" + "=" * 70)
    print("✅ All done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
