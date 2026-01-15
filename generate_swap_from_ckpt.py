# =============================================================================
# swap 可视化生成与解读说明
#
# 本脚本会生成四个图片文件，主要用于理解 latent factor disentanglement：
#
# 1. swap_visualization_inputs.png
#    - 原始输入图片 grid。
#    - 横向（每一列）是一个原始样本（共 n_samples 个，默认8）。
#    - 所有行都是原图。
#
# 2. swap_visualization.png
#    - 所有 latent factor swap 结果的大 grid。
#    - 每一行对应一个 latent factor（如factor 0, 1, 2, ..., 共 latent_unit 个）。
#    - 每一列对应一个原始样本（和输入顺序一致）。
#    - 每个图片表示：将该列 sample 的所有 factor 都来自原图，仅将当前“这一行”的 factor 替换成该行对应的采样值。
#    - 第一列通常可以作为 swap 的 target/reference（因为第一列的 swapping 通常是自身，等价于 src）。
#
# 3. swap_visualization_labeled_pageX.png
#    - 带标签的 swap 可视化（分页），便于清楚看到每个 factor swap 的 effect。
#    - 第一行：原始输入图片（src/source，n_samples个）。
#    - 后续每一行：对应一次 latent factor swap，第i行交换第i-1个factor。
#    - 每列始终是同一个 target 样本。
#    - 第一列始终是原图（src/source），也是每个swap的参照。
#
# 4. analyze_factor_correspondence（函数，不会保存图，但会输出前几个因子的像素变化分析结果）。
#
# 总结：
#   - 所有可视化中，“每列”为同一个 target/original 样本；“每行”为不同 factor 交换结果（第一行是原图）。
#   - 第一列是参考原图（src/source），后续列为不同target。
#   - 如果 factor disentangle 得好，则一行swap只应改变一种物理属性（如颜色、旋转等）。
# =============================================================================

#!/usr/bin/env python
"""
从 checkpoint 生成 sample swap 可视化图片
不需要训练，只需要加载 checkpoint 进行推理
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('/mnt/data_7tb/selena/projects/EncDiff')

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.data.disdata import Shapes3DTrain
from torchvision.utils import make_grid
import os

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
    print(f"\n{'='*70}")
    print("🎨 Generating Swap Visualization")
    print(f"{'='*70}")
    
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
    
    print(f"   Batch shape: {batch_images.shape}")
    print(f"   Latent units: {model.model.diffusion_model.latent_unit}")
    
    # 调用 log_images 生成 swap 样本
    print(f"   Generating swapped samples with DDIM ({ddim_steps} steps)...")
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
    
    print(f"   Generated keys: {list(log.keys())}")
    
    # 保存原始输入
    if 'inputs' in log:
        inputs = log['inputs']
        grid = make_grid(inputs, nrow=4)
        grid = (grid + 1) / 2  # normalize to [0,1]
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 1)
        
        input_path = output_path.replace('.png', '_inputs.png')
        Image.fromarray((grid * 255).astype(np.uint8)).save(input_path)
        print(f"   ✅ Saved inputs: {input_path}")
    
    # 保存 swap 结果
    if 'samples_swapping' in log:
        swapping = log['samples_swapping']
        print(f"   Swapping shape: {swapping.shape}")
        
        # swapping 的形状是 (latent_unit * n_samples, C, H, W)
        # 重塑为 grid
        latent_unit = model.model.diffusion_model.latent_unit
        
        # 保存完整的 swap grid
        grid = make_grid(swapping, nrow=n_samples)
        grid = (grid + 1) / 2
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 1)
        
        Image.fromarray((grid * 255).astype(np.uint8)).save(output_path)
        print(f"   ✅ Saved swap visualization: {output_path}")
        
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
    - 第一行: 原始输入图片（src/source/target，也就是每个样本本身）
    - 后续每行: swap 第 i 个 latent unit 后的结果（每行固定只有该 factor 被换）
    - 每一列：固定为同一个 target（= 第i个输入），左上角即第一个原图样本
    """
    
    # 分页显示（因为有20个latent units太多了）
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
                # swapping 的排列是 [factor0_sample0, factor0_sample1, ..., factor1_sample0, ...]
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
        print(f"   ✅ Saved labeled page {page+1}: {page_path}")


def analyze_factor_correspondence(swapping, inputs, latent_unit, n_samples):
    """
    分析每个 latent factor 对应的物理意义
    通过计算 swap 前后的像素变化来估计
    """
    print(f"\n{'='*70}")
    print("📊 Analyzing Factor Correspondence")
    print(f"{'='*70}")
    
    # 计算每个 factor swap 后的平均变化
    for factor_idx in range(min(latent_unit, 6)):  # 只分析前6个
        changes = []
        for sample_idx in range(1, n_samples):  # 跳过第一个（它是reference）
            img_idx = factor_idx * n_samples + sample_idx
            original = inputs[sample_idx].numpy()
            swapped = swapping[img_idx].numpy()
            change = np.abs(original - swapped).mean()
            changes.append(change)
        
        avg_change = np.mean(changes)
        print(f"   Factor {factor_idx}: avg pixel change = {avg_change:.4f}")


def main():
    print("=" * 70)
    print("🎨 Generate Swap Visualization from Checkpoint")
    print("=" * 70)
    
    # 配置路径
    config_path = '/mnt/data_7tb/selena/projects/EncDiff/configs/latent-diffusion/shapes3d-vq-4-16-encdiff.yaml'
    ckpt_path = '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T07-42-42_shapes3d-vq-4-16-encdiff23/checkpoints/last.ckpt'
    output_dir = '/mnt/data_7tb/selena/projects/EncDiff/logs/2026-01-10T07-42-42_shapes3d-vq-4-16-encdiff23/swap_analysis'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("\n📦 Loading model...")
    model, config = load_model(config_path, ckpt_path)
    print(f"   ✅ Model loaded")
    print(f"   Latent units: {model.model.diffusion_model.latent_unit}")
    
    # 加载数据集
    print("\n📦 Loading dataset...")
    dataset = Shapes3DTrain()
    print(f"   ✅ Dataset size: {len(dataset)}")
    
    # 生成可视化
    output_path = os.path.join(output_dir, 'swap_visualization.png')
    log = generate_swap_visualization(
        model,
        dataset,
        output_path,
        n_samples=8,
        ddim_steps=200
    )
    
    print("\n" + "=" * 70)
    print("✅ Done! Check output in:", output_dir)
    print("=" * 70)
    print("\n💡 如何解读 swap 图片:")
    print("   - 每行对应一个 latent factor (共20个)")
    print("   - 每列对应一个原始样本")
    print("   - swap_visualization_inputs.png 中每张图是原图（src/target）")
    print("   - swap_visualization.png 和 _labeled_pageX.png 各行是某个因子swap的结果，第一行/左上角是原图本身")
    print("   - 第一列总是参照原图 (src/source/target)")
    print("   - 如果某行只改变了一种属性(如颜色)，说明该 factor 学到了该属性")
    print("   - shapes3d 的真实 factors: Floor Hue, Wall Hue, Object Hue, Scale, Shape, Orientation")


if __name__ == "__main__":
    main()
