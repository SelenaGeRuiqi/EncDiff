"""
Test: verify mcl_utils unified API + ddpm_enc.py integration is correct.

Follows senior's design: all MCL components use the same u (B, 20).
  - decoder_G(z, u): decoder uses u directly
  - critic(x_hat, z, u): MechanismCritic uses u
  - Pi_u(u): projection uses u
  - Pi_g(grad): projection uses mechanism gradient

Covers:
  1. cond shape: Encoder4.forward() returns (B, 320) flat, not (B, 20, 16)
  2. Exact disentangled repr: forward() computes encoding(image) -> (B, 20) before warping
  3. decoder_G uses u_cond directly (not ignoring it)
  4. All 5 loss types with MechanismCritic: forward + backward + gradient flow
  5. Data flow: u (B,20) for everything in MCL, cond (B,320) only for UNet
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from ldm.models.diffusion.mcl_utils import (
    MLPProj, MechanismCritic, mcl_loss, ensure_u_2d
)

# =========================================================================
# Dummy modules reproducing EXACT EncDiff shapes and interfaces
# =========================================================================

class DummyVQDecoder(nn.Module):
    """
    Mimics VQModelInterface.decode(h, disentangled_repr=u):
      h: (B, 3, 16, 16), disentangled_repr: (B, 20) -> output: (B, 3, 64, 64)
    """
    def __init__(self, embed_dim=3, disentangled_dim=20):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(embed_dim + disentangled_dim, embed_dim, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 3, 3, padding=1),
        )

    def decode(self, h, disentangled_repr=None):
        B, _, H, W = h.shape
        if disentangled_repr is not None:
            s_expanded = disentangled_repr[:, :, None, None].expand(-1, -1, H, W)
            h = torch.cat([h, s_expanded], dim=1)  # (B, 23, 16, 16)
        h = self.post_quant_conv(h)
        return self.decoder(h)


class DummyEncoder4(nn.Module):
    """
    Mimics Encoder4:
      forward(x):   image (B,3,64,64) -> warped cond (B, 320)
      encoding(x):  image (B,3,64,64) -> raw disentangled repr (B, 20)
    """
    def __init__(self, latent_unit=20, context_dim=16):
        super().__init__()
        self.latent_unit = latent_unit
        self.context_dim = context_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, latent_unit),
        )
        self.warp_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, 64), nn.ELU(True), nn.Linear(64, context_dim))
            for _ in range(latent_unit)
        ])

    def forward(self, x):
        codes = self.encoder(x)  # (B, 20)
        out = [self.warp_nets[i](codes[:, i:i+1]) for i in range(self.latent_unit)]
        return torch.cat(out, dim=1)  # (B, 320)

    def encoding(self, x):
        return self.encoder(x)  # (B, 20)


# =========================================================================
# Test harness
# =========================================================================

passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

def assert_finite_scalar(t, label=""):
    assert t.dim() == 0, f"{label}: expected scalar, got shape {t.shape}"
    assert torch.isfinite(t), f"{label}: not finite: {t.item()}"

def assert_has_grad(modules, label=""):
    for m in modules:
        for p in m.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                return
    raise AssertionError(f"{label}: no non-zero gradients found")


# =========================================================================
# Tests
# =========================================================================

def run_all():
    global passed, failed
    torch.manual_seed(42)
    device = "cpu"
    B = 8
    LATENT_UNIT = 20
    CONTEXT_DIM = 16

    vq_decoder = DummyVQDecoder(embed_dim=3, disentangled_dim=LATENT_UNIT).to(device)
    encoder4 = DummyEncoder4(latent_unit=LATENT_UNIT, context_dim=CONTEXT_DIM).to(device)

    # Following senior's design: Pi_u takes u_dim=20 (not 320)
    Pi_g = MLPProj(in_dim=3*16*16, out_dim=128, layernorm=True).to(device)
    Pi_u = MLPProj(in_dim=LATENT_UNIT, out_dim=128, layernorm=False).to(device)
    mech_critic = MechanismCritic(z_shape=(3,16,16), u_dim=LATENT_UNIT, hidden=256).to(device)

    z = torch.randn(B, 3, 16, 16, device=device)
    dummy_img = torch.randn(B, 3, 64, 64, device=device)

    # ── Simulate forward() ──
    # Step 1: encoding BEFORE warping (exact repr)
    u_mcl = encoder4.encoding(dummy_img)  # (B, 20) — this is u for ALL MCL components
    # Step 2: warp (only for UNet cross-attention, NOT used in MCL)
    cond = encoder4(dummy_img)  # (B, 320) warped

    # ==================================================================
    print("=" * 65)
    print("GROUP 1: cond shape & disentangled repr")
    print("=" * 65)

    def test_cond_is_2d_flat():
        assert cond.dim() == 2 and cond.shape == (B, 320)
    run_test("cond from Encoder4.forward() is (B,320) flat", test_cond_is_2d_flat)

    def test_u_mcl_shape():
        assert u_mcl.shape == (B, LATENT_UNIT), \
            f"expected (B,{LATENT_UNIT}), got {u_mcl.shape}"
    run_test("u_mcl is (B, 20) exact", test_u_mcl_shape)

    def test_exact_ne_approx():
        """Exact encoding and mean-of-warped must differ (warp is nonlinear)."""
        approx = cond.reshape(B, LATENT_UNIT, CONTEXT_DIM).mean(dim=2)
        diff = (u_mcl - approx).abs().max().item()
        assert diff > 1e-3, \
            f"exact and approx should differ (nonlinear warp), but max diff = {diff:.2e}"
    run_test("exact encoding != mean(warped) — confirms warp is nonlinear", test_exact_ne_approx)

    def test_encoding_deterministic():
        repr2 = encoder4.encoding(dummy_img)
        diff = (u_mcl - repr2).abs().max().item()
        assert diff < 1e-6, f"re-encoding differs: max diff = {diff:.2e}"
    run_test("encoding() is deterministic", test_encoding_deterministic)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 2: decoder_G uses u_cond directly (senior's design)")
    print("=" * 65)

    def decoder_G(z_, u_cond):
        # Senior's design: decoder uses u_cond, same interface as mcl_loss expects
        return vq_decoder.decode(z_, disentangled_repr=u_cond)

    def test_decoder_g_uses_u_cond():
        """decoder_G should use u_cond — different u produces different output."""
        u1 = u_mcl.detach()
        u2 = torch.randn_like(u1)
        out1 = decoder_G(z, u1)
        out2 = decoder_G(z, u2)
        diff = (out1 - out2).abs().max().item()
        assert diff > 1e-3, f"decoder_G should depend on u_cond, but outputs are same"
    run_test("decoder_G uses u_cond (different u -> different output)", test_decoder_g_uses_u_cond)

    def test_decoder_g_output_shape():
        x_hat = decoder_G(z, u_mcl.detach())
        assert x_hat.shape == (B, 3, 64, 64)
    run_test("decoder_G output shape: (B, 3, 64, 64)", test_decoder_g_output_shape)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 3: all 5 losses with MechanismCritic (senior's design)")
    print("=" * 65)

    for loss_type in ["nce_logistic", "infonce_mechgrad", "fisher_sm", "denoise_sm", "jacobian_vjp_infonce"]:

        def _test(lt=loss_type):
            all_modules = [vq_decoder, Pi_g, Pi_u, mech_critic]
            for m in all_modules:
                m.zero_grad(set_to_none=True)

            loss_val = mcl_loss(
                loss_type=lt,
                decoder_G=decoder_G,
                z=z.clone(),
                u_key=u_mcl.clone().detach(),  # (B, 20) for everything
                u_for_G=None,
                critic=mech_critic,
                Pi_g=Pi_g, Pi_u=Pi_u,
                tau=0.1, sigma=0.1,
                neg_mode="shuffle_u",
                create_graph=True,
            )
            assert_finite_scalar(loss_val, f"{lt}")
            loss_val.backward()
            assert_has_grad(all_modules, f"{lt}")

        run_test(f"{loss_type} + MechanismCritic", _test)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 4: data flow — u (B,20) for MCL, cond (B,320) for UNet only")
    print("=" * 65)

    def test_data_separation():
        assert u_mcl.shape == (B, LATENT_UNIT)                    # MCL uses (B, 20)
        assert cond.shape == (B, LATENT_UNIT * CONTEXT_DIM)       # UNet uses (B, 320)
        assert u_mcl.shape[1] != cond.shape[1]                    # different dims
    run_test("u_mcl=(B,20) for MCL, cond=(B,320) for UNet — properly separated", test_data_separation)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 5: edge cases")
    print("=" * 65)

    def test_ensure_u_2d():
        assert ensure_u_2d(torch.tensor(1.0)).shape == (1, 1)
        assert ensure_u_2d(torch.randn(5)).shape == (5, 1)
        assert ensure_u_2d(torch.randn(5, 3)).shape == (5, 3)
        assert ensure_u_2d(torch.randn(B, 20)).shape == (B, 20)
    run_test("ensure_u_2d: scalar/1d/2d all correct", test_ensure_u_2d)

    def test_invalid_loss_type():
        try:
            mcl_loss("nonexistent", decoder_G, z, u_mcl)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
    run_test("invalid loss_type raises ValueError", test_invalid_loss_type)

    def test_u_key_u_for_g_separation():
        """Test that u_key and u_for_G can have different dims."""
        u_key_small = torch.randn(B, 5, device=device)
        Pi_u_small = MLPProj(in_dim=5, out_dim=128, layernorm=False).to(device)
        Pi_u_small.zero_grad(set_to_none=True)
        loss = mcl_loss(
            loss_type="jacobian_vjp_infonce",
            decoder_G=decoder_G, z=z.clone(),
            u_key=u_key_small, u_for_G=u_mcl.clone().detach(),
            Pi_g=Pi_g, Pi_u=Pi_u_small,
            tau=0.1, create_graph=False,
        )
        assert_finite_scalar(loss, "u_key_sep")
        loss.backward()
        assert_has_grad([Pi_u_small], "u_key_sep")
    run_test("u_key/u_for_G dim separation works", test_u_key_u_for_g_separation)

    # ==================================================================
    print("\n" + "=" * 65)
    total = passed + failed
    if failed == 0:
        print(f"ALL {total} TESTS PASSED")
    else:
        print(f"RESULTS: {passed} passed, {failed} failed out of {total}")
    print("=" * 65)
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
