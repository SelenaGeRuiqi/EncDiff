# mcl_utils.py
# Unified MCL (Mechanism Contrastive Learning) with 5 objective function variants.
# loss_type: {"nce_logistic","infonce_mechgrad","fisher_sm","denoise_sm","jacobian_vjp_infonce"}
# References:
#   - NCE: Gutmann & Hyvärinen (2010) https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf
#   - Score matching: Hyvärinen (2019) https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_u_2d(u: torch.Tensor) -> torch.Tensor:
    if u.dim() == 0:
        return u.view(1, 1)
    if u.dim() == 1:
        return u.unsqueeze(1)
    return u


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def info_nce_from_qk(q: torch.Tensor, k: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    logits = (q @ k.t()) / (tau + 1e-12)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def hutchinson_divergence(z: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(z)
    inner = (score * eps).sum()
    grad = torch.autograd.grad(inner, z, create_graph=True)[0]
    return (grad * eps).flatten(1).sum(dim=1)


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim=128, layernorm=False):
        super().__init__()
        layers = []
        if layernorm:
            layers.append(nn.LayerNorm(in_dim))
        layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MechanismCritic(nn.Module):
    def __init__(self, z_shape=(3, 16, 16), u_dim=20, hidden=256):
        super().__init__()
        zc, zh, zw = z_shape
        self.img = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.z_fc = nn.Linear(zc * zh * zw, hidden)
        self.u_fc = nn.Linear(u_dim, hidden)
        self.out = nn.Sequential(nn.ReLU(True), nn.Linear(hidden, 1))

    def forward(self, x_hat, z, u):
        img_feat = self.img(x_hat).flatten(1)
        z_feat = self.z_fc(z.flatten(1))
        u_feat = self.u_fc(u)
        if img_feat.size(1) < z_feat.size(1):
            img_feat = F.pad(img_feat, (0, z_feat.size(1) - img_feat.size(1)), value=0.0)
        else:
            img_feat = img_feat[:, : z_feat.size(1)]
        h = z_feat + u_feat + img_feat
        return self.out(h).squeeze(1)


def mcl_loss(loss_type: str,
             decoder_G,
             z: torch.Tensor,
             u_key: torch.Tensor,
             u_for_G: torch.Tensor | None = None,
             critic: nn.Module | None = None,
             Pi_g: nn.Module | None = None,
             Pi_u: nn.Module | None = None,
             tau: float = 0.1,
             sigma: float = 0.1,
             neg_mode: str = "shuffle_u",
             create_graph: bool = False) -> torch.Tensor:
    """
    Unified API for mcl loss candidates.

    Inputs:
      - loss_type: {"nce_logistic","infonce_mechgrad","fisher_sm","denoise_sm","jacobian_vjp_infonce"}
      - decoder_G: callable (z, uG) -> x_hat
      - z: latent tensor (B, ...)
      - u_key: auxiliary variable used as contrastive key (scalar or vector); only used when Pi_u is needed
      - u_for_G: conditioning actually fed into decoder/critic; if None, uses u_key
      - critic: required for {"nce_logistic","infonce_mechgrad","fisher_sm","denoise_sm"}
      - Pi_g, Pi_u: required for {"infonce_mechgrad","jacobian_vjp_infonce"}
      - tau: InfoNCE temperature
      - sigma: noise level for denoising score matching
      - neg_mode: {"shuffle_u","shuffle_z"} for NCE
      - create_graph: whether to create higher-order graph for z-grad features

    Output:
      - scalar loss (torch.Tensor)
    """
    u = ensure_u_2d(u_key)
    uG = ensure_u_2d(u_for_G) if u_for_G is not None else u

    if loss_type == "nce_logistic":
        if critic is None:
            raise ValueError("critic is required for nce_logistic")
        x_pos = decoder_G(z, uG)
        logit_pos = critic(x_pos, z, uG)

        if neg_mode == "shuffle_u":
            perm = torch.randperm(uG.size(0), device=uG.device)
            u_neg = uG[perm]
            z_neg = z
        elif neg_mode == "shuffle_z":
            perm = torch.randperm(uG.size(0), device=uG.device)
            z_neg = z[perm]
            u_neg = uG
        else:
            raise ValueError("neg_mode must be 'shuffle_u' or 'shuffle_z'")

        x_neg = decoder_G(z_neg, u_neg)
        logit_neg = critic(x_neg, z_neg, u_neg)

        loss_pos = F.binary_cross_entropy_with_logits(logit_pos, torch.ones_like(logit_pos))
        loss_neg = F.binary_cross_entropy_with_logits(logit_neg, torch.zeros_like(logit_neg))
        return loss_pos + loss_neg

    if loss_type == "infonce_mechgrad":
        if critic is None or Pi_g is None or Pi_u is None:
            raise ValueError("critic, Pi_g, Pi_u are required for infonce_mechgrad")
        z_ = z.requires_grad_(True)
        x_hat = decoder_G(z_, uG)
        s = critic(x_hat, z_, uG)
        g = torch.autograd.grad(s.sum(), z_, create_graph=create_graph)[0]
        q = l2norm(Pi_g(g.flatten(1)))
        k = l2norm(Pi_u(u))
        return info_nce_from_qk(q, k, tau=tau)

    if loss_type == "fisher_sm":
        if critic is None:
            raise ValueError("critic is required for fisher_sm")
        z_ = z.requires_grad_(True)
        x_hat = decoder_G(z_, uG)
        s = critic(x_hat, z_, uG)
        score = torch.autograd.grad(s.sum(), z_, create_graph=True)[0]
        score_norm = 0.5 * score.flatten(1).pow(2).sum(dim=1)
        div = hutchinson_divergence(z_, score)
        return (score_norm + div).mean()

    if loss_type == "denoise_sm":
        if critic is None:
            raise ValueError("critic is required for denoise_sm")
        eps = torch.randn_like(z)
        z_t = (z + sigma * eps).requires_grad_(True)
        x_hat = decoder_G(z_t, uG)
        s = critic(x_hat, z_t, uG)
        score = torch.autograd.grad(s.sum(), z_t, create_graph=True)[0]
        target = -(eps / (sigma + 1e-12))
        return (score - target).flatten(1).pow(2).mean()

    if loss_type == "jacobian_vjp_infonce":
        if Pi_g is None or Pi_u is None:
            raise ValueError("Pi_g, Pi_u are required for jacobian_vjp_infonce")
        z_ = z.requires_grad_(True)
        x_hat = decoder_G(z_, uG)
        v = torch.randn_like(x_hat)
        scalar = (x_hat * v).sum()
        mechfeat = torch.autograd.grad(scalar, z_, create_graph=create_graph)[0]
        q = l2norm(Pi_g(mechfeat.flatten(1)))
        k = l2norm(Pi_u(u))
        return info_nce_from_qk(q, k, tau=tau)

    raise ValueError(f"Unknown loss_type: {loss_type}")
