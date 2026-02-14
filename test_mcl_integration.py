"""
Test: verify mcl_utils unified API works correctly in EncDiff's data flow.

Simulates the exact shapes and calling convention used in ddpm_enc.py:
  - z (latent code): (B, 3, 16, 16)
  - u (concept repr): (B, 20, 16) -> flattened to (B, 320)
  - decoder_G(z, u_flat) -> x_hat (B, 3, 64, 64)
  - concept_encoder(x_hat) -> u_hat (B, 320)

Tests:
  1. All 5 loss types produce finite scalar losses
  2. All 5 loss types allow backward() without error
  3. ConceptEncoderCritic matches the old mech_score_mse semantics
  4. MechanismCritic works as a standalone trainable critic
  5. Both critic modes work with all compatible loss types
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from ldm.models.diffusion.mcl_utils import (
    MLPProj, MechanismCritic, ConceptEncoderCritic, mcl_loss, ensure_u_2d
)

# ---- Dummy modules matching EncDiff shapes ----

class DummyDecoder(nn.Module):
    """Mimics differentiable_decode_first_stage: (B,3,16,16) + (B,320) -> (B,3,64,64)"""
    def __init__(self):
        super().__init__()
        self.proj_u = nn.Linear(320, 3)
        self.up = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, z, u):
        bias = self.proj_u(u).view(-1, 3, 1, 1)
        return self.up(z + bias)


class DummyConceptEncoder(nn.Module):
    """Mimics get_learned_conditioning: (B,3,64,64) -> (B,320)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 320)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def make_modules(device):
    decoder = DummyDecoder().to(device)
    concept_enc = DummyConceptEncoder().to(device)
    Pi_g = MLPProj(in_dim=3 * 16 * 16, out_dim=128, layernorm=True).to(device)
    Pi_u = MLPProj(in_dim=320, out_dim=128, layernorm=False).to(device)
    mech_critic = MechanismCritic(z_shape=(3, 16, 16), u_dim=320, hidden=256).to(device)
    concept_critic = ConceptEncoderCritic(concept_enc)
    return decoder, concept_enc, Pi_g, Pi_u, mech_critic, concept_critic


def test_loss(name, loss_tensor, modules_to_check):
    """Check loss is finite scalar, then backward and check grads exist."""
    assert loss_tensor.dim() == 0, f"{name}: loss should be scalar, got shape {loss_tensor.shape}"
    assert torch.isfinite(loss_tensor), f"{name}: loss is not finite: {loss_tensor.item()}"

    # zero grads
    for m in modules_to_check:
        if isinstance(m, nn.Module):
            m.zero_grad(set_to_none=True)

    loss_tensor.backward()

    has_grad = False
    for m in modules_to_check:
        if isinstance(m, nn.Module):
            for p in m.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
                    break
    assert has_grad, f"{name}: no gradients found after backward"
    print(f"  PASS  {name}: loss={loss_tensor.item():.4f}")


def run_tests():
    torch.manual_seed(42)
    device = "cpu"
    B = 8

    z = torch.randn(B, 3, 16, 16, device=device)
    u_flat = torch.randn(B, 320, device=device)  # flattened (B, 20, 16)

    decoder, concept_enc, Pi_g, Pi_u, mech_critic, concept_critic = make_modules(device)
    decoder_G = lambda z_, u_: decoder(z_, u_)

    passed = 0
    failed = 0

    # ============================================================
    # Test 1: ConceptEncoderCritic semantic equivalence
    # ============================================================
    print("=" * 60)
    print("Test 1: ConceptEncoderCritic matches old mech_score_mse")
    print("=" * 60)
    try:
        x_test = decoder_G(z, u_flat)
        u_hat = concept_enc(x_test)
        old_score = -((u_hat - u_flat) ** 2).sum(dim=1)
        new_score = concept_critic(x_test, z, u_flat)
        diff = (old_score - new_score).abs().max().item()
        assert diff < 1e-6, f"Score mismatch: max diff={diff}"
        print(f"  PASS  max diff = {diff:.2e}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {e}")
        failed += 1

    # ============================================================
    # Test 2: All 5 loss types with ConceptEncoderCritic
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 2: All 5 loss types with ConceptEncoderCritic")
    print("=" * 60)

    loss_configs = [
        ("nce_logistic", dict(critic=concept_critic)),
        ("infonce_mechgrad", dict(critic=concept_critic, Pi_g=Pi_g, Pi_u=Pi_u, create_graph=False)),
        ("fisher_sm", dict(critic=concept_critic)),
        ("denoise_sm", dict(critic=concept_critic, sigma=0.1)),
        ("jacobian_vjp_infonce", dict(Pi_g=Pi_g, Pi_u=Pi_u, create_graph=False)),
    ]

    for loss_type, kwargs in loss_configs:
        try:
            loss = mcl_loss(
                loss_type=loss_type,
                decoder_G=decoder_G,
                z=z.clone(),
                u_key=u_flat.clone(),
                tau=0.1,
                **kwargs,
            )
            modules = [decoder, concept_enc, Pi_g, Pi_u]
            test_loss(f"concept_encoder + {loss_type}", loss, modules)
            passed += 1
        except Exception as e:
            print(f"  FAIL  concept_encoder + {loss_type}: {e}")
            failed += 1

    # ============================================================
    # Test 3: All 5 loss types with MechanismCritic
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 3: All 5 loss types with MechanismCritic")
    print("=" * 60)

    loss_configs_mc = [
        ("nce_logistic", dict(critic=mech_critic)),
        ("infonce_mechgrad", dict(critic=mech_critic, Pi_g=Pi_g, Pi_u=Pi_u, create_graph=False)),
        ("fisher_sm", dict(critic=mech_critic)),
        ("denoise_sm", dict(critic=mech_critic, sigma=0.1)),
        ("jacobian_vjp_infonce", dict(Pi_g=Pi_g, Pi_u=Pi_u, create_graph=False)),
    ]

    for loss_type, kwargs in loss_configs_mc:
        try:
            loss = mcl_loss(
                loss_type=loss_type,
                decoder_G=decoder_G,
                z=z.clone(),
                u_key=u_flat.clone(),
                tau=0.1,
                **kwargs,
            )
            modules = [decoder, mech_critic, Pi_g, Pi_u]
            test_loss(f"mechanism_critic + {loss_type}", loss, modules)
            passed += 1
        except Exception as e:
            print(f"  FAIL  mechanism_critic + {loss_type}: {e}")
            failed += 1

    # ============================================================
    # Test 4: ensure_u_2d correctness
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 4: ensure_u_2d shape handling")
    print("=" * 60)
    try:
        assert ensure_u_2d(torch.tensor(1.0)).shape == (1, 1)
        assert ensure_u_2d(torch.randn(5)).shape == (5, 1)
        assert ensure_u_2d(torch.randn(5, 3)).shape == (5, 3)
        print("  PASS  scalar->（1,1), 1d->(N,1), 2d unchanged")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {e}")
        failed += 1

    # ============================================================
    # Test 5: u_for_G separation (key vs. conditioning)
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 5: u_key vs u_for_G separation")
    print("=" * 60)
    try:
        u_key_small = torch.randn(B, 10, device=device)
        Pi_u_small = MLPProj(in_dim=10, out_dim=128, layernorm=False).to(device)

        loss = mcl_loss(
            loss_type="jacobian_vjp_infonce",
            decoder_G=decoder_G,
            z=z.clone(),
            u_key=u_key_small,
            u_for_G=u_flat.clone(),
            Pi_g=Pi_g,
            Pi_u=Pi_u_small,
            tau=0.1,
            create_graph=False,
        )
        test_loss("u_key(10d) vs u_for_G(320d)", loss, [decoder, Pi_g, Pi_u_small])
        passed += 1
    except Exception as e:
        print(f"  FAIL  {e}")
        failed += 1

    # ============================================================
    # Test 6: Invalid loss_type raises ValueError
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 6: Invalid loss_type raises ValueError")
    print("=" * 60)
    try:
        mcl_loss("nonexistent", decoder_G, z, u_flat)
        print("  FAIL  should have raised ValueError")
        failed += 1
    except ValueError as e:
        print(f"  PASS  caught: {e}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  wrong exception: {e}")
        failed += 1

    # ============================================================
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
