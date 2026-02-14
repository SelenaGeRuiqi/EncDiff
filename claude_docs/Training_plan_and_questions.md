## 问题：
- implementation和实验计划有无问题
- 所有模块共用 lr=2e-7。MCL 新模块（Pi_g, Pi_u, MechanismCritic）从随机初始化训起，是否需要更高的 lr？

## Experiment： run_mcl_parallel_amd.sh
plan: 在之前训练好的concat EncDiff（VQ-VAE concat + diffusion，无 MCL）的基础上引入MCL微调3个epoch(先看看MCL loss 在不在降，5 种 loss type 哪个表现好，λ=0.01 和 0.05 哪个更合适，然后再后续调整跑更多的)
数据集：MPI3D（先跑这个，看效果再跑 Shapes3D）
从 checkpoint 恢复：MPI3D 的 concat EncDiff checkpoint（strict=False，Pi_g/Pi_u/MechanismCritic 随机初始化）
GPU 分配：8 卡，第一次训练分两轮。Round 1 跑 8 个（5×λ=0.01 + 3×λ=0.05），Round 2 跑剩余 2 个。瞅瞅结果咋样发给minghao之后再接着跑更多的。
训练配置：max_epochs: 3，batch_size: 128，lr: 2e-7，WandB 记录 loss_mcl, loss_simple, mcl_diffusion_ratio（监控是否有两个obj梯度collapse）, factor_vae_score, dci_disentanglement


## Implementation：
1. ldm/models/diffusion/mcl_utils.py — 搬运test_5_mcl.py原始代码，包含 mcl_loss 5种、MechanismCritic、MLPProj 等。
2. ldm/models/diffusion/ddpm_enc.py — 主要改了这些：
（1） __init__  
- 新增参数 lambda_mcl, mcl_type, use_mcl 等  
- 初始化 Pi_g (768→128), Pi_u (20→128), MechanismCritic (z=(3,16,16), u=20)

（2） forward  
- 在 warping 前调用 Encoder4.encoding(c) 得到 u=(B,20)，存到 self._mcl_disentangled_repr

（3） p_losses  
- 构造 decoder_G(z, u)=frozen_VQ-VAE.decode(z, u)
- 用 mcl_loss() 计算 MCL
- 总 loss = diffusion_loss + lambda_mcl * MCL

（4） configure_optimizers  
- 把 Pi_g, Pi_u, MechanismCritic 的参数加到 optimizer

  mcl_loss(
      loss_type=self.mcl_type,       # config 里指定，5 选 1
      decoder_G=decoder_G,           # frozen VQ-VAE decode(z, u)
      z=x_start,                     # clean VQ-VAE latent (B, 3, 16, 16)
      u_key=u_mcl,                   # Encoder4.encoding(image) (B, 20)
      u_for_G=None,                  # 用 u_key 同时做 decoder 和 contrastive 的 u
      critic=self.mcl_critic,        # MechanismCritic
      Pi_g=self.Pi_g,                # MLPProj(768→128)
      Pi_u=self.Pi_u,                # MLPProj(20→128)
      tau=0.1, sigma=0.1,
      neg_mode="shuffle_u",
      create_graph=True,
  )