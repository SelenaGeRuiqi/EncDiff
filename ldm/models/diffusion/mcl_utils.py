import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim=128, layernorm=False):
        super().__init__()
        layers = []
        if layernorm:
            layers.append(nn.LayerNorm(in_dim))
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# u = encoder(x), u_hat = encoder(x_hat)
def mech_score_mse(u_hat, u): # Gaussian -> MSE
    return -((u_hat - u) ** 2).sum(dim=1)

def mech_grad_g(decoder_G, concept_encoder, z, u,
                create_graph=False, normalize_g=True):
    z = z.requires_grad_(True)
    x_hat = decoder_G(z, u)
    u_hat = concept_encoder(x_hat)
    score = mech_score_mse(u_hat, u)
    g = torch.autograd.grad(score.sum(), z, create_graph=create_graph)[0]
    if normalize_g:
        denom = g.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8
        g = g / denom
    return g


def mcl_infonce_loss(decoder_G, concept_encoder, Pi_g, Pi_u,
                     z, u, tau=0.1, create_graph_mcl=False): 
    g = mech_grad_g(decoder_G, concept_encoder, z, u,
                    create_graph=create_graph_mcl, normalize_g=True)
    print("mechfeat shape:", g.shape)
    g_flat = g.flatten(1)
    q = F.normalize(Pi_g(g_flat), dim=1)
    k = F.normalize(Pi_u(u), dim=1)
    logits = (q @ k.t()) / (tau + 1e-12)
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)

if __name__ == "__main__":
    # Usage: use your existing encoder and decoder as-is, and add `mcl_infonce_loss`
    # to the final training objective with a weighting coefficient `lambda_mcl`.
    # It is strongly recommended to start from a pretrained model (e.g., on the concatenation of (z, disentangled_repr))
    # and then fine-tune with the MCL loss term added. The weight `lambda_mcl` should
    # be tuned manually for performance.
    torch.manual_seed(0)
    B = 128

    x = torch.randn(B, 3, 64, 64)
    z = torch.randn(B, 3, 16, 16)

    class DummyConceptEncoder(nn.Module):
        def __init__(self, u_dim=20):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(128, u_dim)

        def forward(self, x):
            return self.fc(self.net(x).flatten(1))

    class DummyDecoder(nn.Module):
        def __init__(self, u_dim=20):
            super().__init__()
            self.s_proj = nn.Linear(u_dim, 3)
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(64, 3, 3, padding=1),
            )

        def forward(self, z, u):
            bias = self.s_proj(u).view(-1, 3, 1, 1)
            return self.net(z + bias)

    concept_encoder = DummyConceptEncoder(u_dim=20)
    decoder = DummyDecoder(u_dim=20)

    u = concept_encoder(x)

    decoder_G = lambda z_, u_: decoder(z_, u_)

    Pi_g = MLPProj(in_dim=3 * 16 * 16, out_dim=128, layernorm=True)
    Pi_u = MLPProj(in_dim=20, out_dim=128, layernorm=False)

    loss_mcl = mcl_infonce_loss(
        decoder_G, concept_encoder, Pi_g, Pi_u,
        z=z, u=u, tau=0.1, create_graph_mcl=False
    )

    print("mcl_loss:", float(loss_mcl))

    loss_mcl.backward()
    print("backward ok; mean |grad Pi_g|:",
          Pi_g.net[1].weight.grad.abs().mean().item())