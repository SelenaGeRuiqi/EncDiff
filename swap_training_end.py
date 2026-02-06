from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class SwapVisualizationCallback(Callback):
    """
    每个epoch结束时生成swap可视化
    - 保存到本地experiment目录
    - 上传到WandB
    """
    
    def __init__(self, n_samples=8, ddim_steps=200, 
                 save_locally=True, log_to_wandb=True,
                 factors_per_page=10):
        super().__init__()
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.save_locally = save_locally
        self.log_to_wandb = log_to_wandb
        self.factors_per_page = factors_per_page
        self.fixed_indices = None
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Validation epoch结束时生成swap可视化"""
        
        if trainer.global_rank != 0:
            return
        
        epoch = trainer.current_epoch
        print(f"\n{'='*70}")
        print(f"🎨 Generating Swap Visualization (Epoch {epoch})")
        print(f"{'='*70}")
        
        try:
            val_dataloader = trainer.val_dataloaders[0]
            dataset = val_dataloader.dataset
            
            # 固定样本索引（每次使用相同样本便于对比）
            if self.fixed_indices is None:
                np.random.seed(42)
                self.fixed_indices = np.random.choice(
                    len(dataset), self.n_samples, replace=False
                )
                print(f"   Fixed indices: {self.fixed_indices.tolist()}")
            
            # 收集样本
            images = []
            for idx in self.fixed_indices:
                sample = dataset[idx]
                if torch.is_tensor(sample['image']):
                    img = sample['image']
                else:
                    img = torch.from_numpy(sample['image']).float()
                images.append(img)
            
            batch_images = torch.stack(images, dim=0).to(pl_module.device)
            batch_dict = {'image': batch_images}
            
            # 生成swap可视化
            print(f"   Generating swapped samples (DDIM {self.ddim_steps} steps)...")
            pl_module.eval()
            with torch.no_grad():
                log = pl_module.log_images(
                    batch_dict,
                    N=self.n_samples,
                    sample_swap=True,
                    ddim_steps=self.ddim_steps,
                    ddim_eta=0.0,
                    inpaint=False,
                    plot_progressive_rows=False,
                    plot_diffusion_rows=False
                )
            pl_module.train()
            
            if 'samples_swapping' not in log:
                print(f"   ⚠️  No swap samples generated")
                return
            
            swapping = log['samples_swapping']
            inputs = log.get('inputs', None)
            latent_unit = pl_module.model.diffusion_model.latent_unit
            
            print(f"   Swapping shape: {swapping.shape}")
            print(f"   Latent units: {latent_unit}")
            
            # 保存到本地
            if self.save_locally:
                output_dir = os.path.join(
                    trainer.log_dir,
                    f'swap_epoch_{epoch:03d}'
                )
                os.makedirs(output_dir, exist_ok=True)
                self._save_visualizations(
                    inputs, swapping, latent_unit, output_dir
                )
                print(f"   ✅ Saved locally: {output_dir}")
            
            # 上传到WandB
            if self.log_to_wandb and wandb.run is not None:
                self._log_to_wandb(
                    inputs, swapping, latent_unit, epoch
                )
                print(f"   ✅ Logged to WandB")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*70}\n")
    
    def _save_visualizations(self, inputs, swapping, latent_unit, output_dir):
        """保存所有可视化（按照你原有代码的逻辑）"""
        
        # 1. swap_visualization_inputs.png - 原始输入
        if inputs is not None:
            grid = make_grid(inputs, nrow=4)
            grid = (grid + 1) / 2
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = np.clip(grid, 0, 1)
            Image.fromarray((grid * 255).astype(np.uint8)).save(
                os.path.join(output_dir, 'swap_visualization_inputs.png')
            )
        
        # 2. swap_visualization.png - 完整swap grid
        grid = make_grid(swapping, nrow=self.n_samples)
        grid = (grid + 1) / 2
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 1)
        Image.fromarray((grid * 255).astype(np.uint8)).save(
            os.path.join(output_dir, 'swap_visualization.png')
        )
        
        # 3. swap_visualization_labeled_pageX.png - 带标签分页可视化
        self._create_labeled_pages(
            inputs.cpu() if inputs is not None else None,
            swapping.cpu(),
            latent_unit,
            output_dir
        )
    
    def _create_labeled_pages(self, inputs, swapping, latent_unit, output_dir):
        """
        创建带标签的分页可视化
        完全按照你原有代码的布局：
        - 第一行: 原始输入（src/target）
        - 后续每行: swap第i个latent factor的结果
        - 每列: 同一个target样本
        """
        n_pages = (latent_unit + self.factors_per_page - 1) // self.factors_per_page
        
        for page in range(n_pages):
            start_factor = page * self.factors_per_page
            end_factor = min((page + 1) * self.factors_per_page, latent_unit)
            n_factors_this_page = end_factor - start_factor
            
            fig, axes = plt.subplots(
                n_factors_this_page + 1, self.n_samples,
                figsize=(2*self.n_samples, 2*(n_factors_this_page+1))
            )
            
            # 确保axes是2D
            if n_factors_this_page + 1 == 1:
                axes = axes.reshape(1, -1)
            
            # 第一行: 原始输入
            if inputs is not None:
                for i in range(self.n_samples):
                    img = inputs[i].permute(1, 2, 0).numpy()
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    axes[0, i].imshow(img)
                    axes[0, i].axis('off')
                    if i == 0:
                        axes[0, i].set_ylabel(
                            'Input', fontsize=10,
                            rotation=0, ha='right', labelpad=30
                        )
            
            # 后续行: swap结果
            for factor_idx in range(start_factor, end_factor):
                row_idx = factor_idx - start_factor + 1
                for sample_idx in range(self.n_samples):
                    img_idx = factor_idx * self.n_samples + sample_idx
                    img = swapping[img_idx].permute(1, 2, 0).numpy()
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    axes[row_idx, sample_idx].imshow(img)
                    axes[row_idx, sample_idx].axis('off')
                
                axes[row_idx, 0].set_ylabel(
                    f'Factor {factor_idx}', fontsize=10,
                    rotation=0, ha='right', labelpad=30
                )
            
            plt.tight_layout()
            page_path = os.path.join(
                output_dir,
                f'swap_visualization_labeled_page{page+1}.png'
            )
            plt.savefig(page_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def _log_to_wandb(self, inputs, swapping, latent_unit, epoch):
        """上传到WandB"""
        
        wandb_images = []
        
        # 1. 原始输入
        if inputs is not None:
            grid = make_grid(inputs, nrow=4)
            grid = (grid + 1) / 2
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = np.clip(grid, 0, 1)
            wandb_images.append(
                wandb.Image(grid, caption=f"Epoch {epoch} - Inputs")
            )
        
        # 2. 完整swap grid
        grid = make_grid(swapping, nrow=self.n_samples)
        grid = (grid + 1) / 2
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = np.clip(grid, 0, 1)
        wandb_images.append(
            wandb.Image(grid, caption=f"Epoch {epoch} - Full Swap")
        )
        
        # 3. 每个factor的swap（前10个）
        n_factors_to_log = min(10, latent_unit)
        for factor_idx in range(n_factors_to_log):
            factor_images = swapping[
                factor_idx * self.n_samples : (factor_idx + 1) * self.n_samples
            ]
            grid = make_grid(factor_images, nrow=self.n_samples)
            grid = (grid + 1) / 2
            grid = grid.permute(1, 2, 0).cpu().numpy()
            grid = np.clip(grid, 0, 1)
            wandb_images.append(
                wandb.Image(grid, caption=f"Epoch {epoch} - Factor {factor_idx}")
            )
        
        # 记录到WandB
        wandb.log({
            "swap_visualization": wandb_images,
            "epoch": epoch
        }, step=epoch)
