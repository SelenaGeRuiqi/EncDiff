## Pipeline:
 Image ──→ Encoder4.encoding() ──→ u (B, 20) ──→ MCL
        └─→ Encoder4.forward()  ──→ cond (B, 320) ──→ UNet cross-attn

  p_losses:
    total_loss = diffusion_loss + lambda_mcl * mcl_loss(
        decoder_G = frozen_VQ-VAE.decode(z, u),
        z = x_start (B, 3, 16, 16),
        u = (B, 20),
        critic = MechanismCritic,
        Pi_g, Pi_u

## Dimension Table
- z: x_start,  (B, 3, 16, 16), VQ-VAE 编码后的 clean latent
- u: self._mcl_disentangled_repr, (B, 20), Encoder4.encoding(image)，原始 disentangled 表示
- x_hat: decoder_G(z, u) 的返回值, (B, 3, 64, 64), VQ-VAE decode 出的重建图像

- decoder_G: differentiable_decode_first_stage(z, disentangled_repr=u), (z, u) -> x_hat, frozen VQ-VAE decoder
- critic: self.mcl_critic = MechanismCritic, (x_hat, z, u) -> scalar, MechanismCritic，独立网络
- Pi_g: self.Pi_g = MLPProj(in_dim=768), (B, 768) -> (B, 128), 投影 mechanism gradient
- Pi_u: self.Pi_u = MLPProj(in_dim=20), (B, 20) -> (B, 128), 投影 u

## 在 WandB 上看什么

  MCL 微调的目标是：disentanglement 提升，generation 不崩。所以：

  看涨的指标（MCL 在起作用）：
  - val/factor_vae_score ↑
  - val/dci_disentanglement ↑

  看稳的指标（generation 没崩）：
  - train/loss_simple 不应该明显上升。如果大幅上升，说明 λ 太大，MCL 在破坏生成

  看收敛的指标（MCL 本身在学）：
  - train/loss_mcl 应该下降
  - train/grad_norm_mcl 应该非零且稳定（如果趋近 0 说明 MCL 没在学）

  什么时候停

  实际操作：不需要提前决定 epoch 数。 你的 ModelCheckpoint 会自动保存最佳 checkpoint。你可以先设 max_epochs: 3 跑完看曲线：

  1. 如果 disentanglement 还在涨、loss_simple 稳定 — 说明 3 epoch 不够，改成 5 或 10 继续跑
  2. 如果 disentanglement 涨了然后开始掉、loss_simple 开始涨 — 说明过拟合了，用 best_vae 或 best_dci checkpoint 就是最优点
  3. 如果 disentanglement 完全没动、loss_mcl 没降 — 说明 lr 太小或 λ 太小，MCL 没起作用

  第三种情况最可能发生（因为 lr=2e-7 对随机初始化的模块可能太小），到时候调 lr 或 λ 再跑。

  建议先 3 epoch 探索

  3 epoch 的目的不是训到最优，而是快速看到：
  - MCL loss 在不在降？
  - 5 种 loss type 哪个表现好？
  - λ=0.01 和 0.05 哪个更合适？

  根据这一轮结果，挑最好的 1-2 个组合，再跑更长时间。