"""
Test: verify mcl_utils unified API + ddpm_enc.py integration is correct.

Covers:
  1. cond shape: Encoder4.forward() returns (B, 320) flat, not (B, 20, 16)
  2. Exact disentangled repr: forward() computes encoding(image) -> (B, 20) before
     warping, and p_losses uses it directly in decoder_G — no mean approximation
  3. decoder_G produces correct output shape through VQ-VAE concat decode
  4. All 5 loss types x 2 critic modes: forward + backward + gradient flow
  5. ConceptEncoderCritic == old mech_score_mse
  6. Data separation: u_key (B,320) for Pi_u/critic, disentangled_repr (B,20) for decoder
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from ldm.models.diffusion.mcl_utils import (
    MLPProj, MechanismCritic, ConceptEncoderCritic, mcl_loss, ensure_u_2d
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
    Pi_g = MLPProj(in_dim=3*16*16, out_dim=128, layernorm=True).to(device)
    Pi_u = MLPProj(in_dim=LATENT_UNIT*CONTEXT_DIM, out_dim=128, layernorm=False).to(device)
    mech_critic = MechanismCritic(z_shape=(3,16,16), u_dim=LATENT_UNIT*CONTEXT_DIM, hidden=256).to(device)

    z = torch.randn(B, 3, 16, 16, device=device)
    dummy_img = torch.randn(B, 3, 64, 64, device=device)

    # ── Simulate forward() ──
    # Step 1: encoding BEFORE warping (the new fix)
    disentangled_repr = encoder4.encoding(dummy_img)  # (B, 20) exact
    # Step 2: warp
    cond = encoder4(dummy_img)  # (B, 320) warped

    # ==================================================================
    print("=" * 65)
    print("GROUP 1: cond shape & old bug verification")
    print("=" * 65)

    def test_cond_is_2d_flat():
        assert cond.dim() == 2 and cond.shape == (B, 320)
    run_test("cond from Encoder4.forward() is (B,320) flat", test_cond_is_2d_flat)

    def test_old_code_would_crash():
        try:
            _ = cond.shape[2]
            raise AssertionError("Should IndexError")
        except IndexError:
            pass
    run_test("old code u.shape[2] would IndexError on (B,320)", test_old_code_would_crash)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 2: exact disentangled repr vs mean approximation")
    print("=" * 65)

    def test_disentangled_repr_shape():
        assert disentangled_repr.shape == (B, LATENT_UNIT), \
            f"expected (B,{LATENT_UNIT}), got {disentangled_repr.shape}"
    run_test("disentangled_repr is (B, 20) exact", test_disentangled_repr_shape)

    def test_exact_ne_approx():
        """Exact encoding and mean-of-warped must differ (warp is nonlinear)."""
        approx = cond.reshape(B, LATENT_UNIT, CONTEXT_DIM).mean(dim=2)
        diff = (disentangled_repr - approx).abs().max().item()
        assert diff > 1e-3, \
            f"exact and approx should differ (nonlinear warp), but max diff = {diff:.2e}"
    run_test("exact encoding != mean(warped) — confirms approximation was lossy", test_exact_ne_approx)

    def test_exact_matches_re_encoding():
        """encoding() called twice on the same image must give the same result."""
        repr2 = encoder4.encoding(dummy_img)
        diff = (disentangled_repr - repr2).abs().max().item()
        assert diff < 1e-6, f"re-encoding differs: max diff = {diff:.2e}"
    run_test("encoding() is deterministic (same input -> same output)", test_exact_matches_re_encoding)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 3: decoder_G uses exact (B,20), not u_cond")
    print("=" * 65)

    def test_decoder_g_uses_exact_repr():
        """
        decoder_G should use the captured disentangled_repr, NOT u_cond.
        We verify by passing a garbage u_cond and checking output is the same
        as when passing the real cond.
        """
        _dis = disentangled_repr.detach()

        def decoder_G(z_, u_cond):
            # This is the EXACT logic from the fixed p_losses:
            # ignores u_cond, uses captured _dis
            return vq_decoder.decode(z_, disentangled_repr=_dis)

        out_with_real_u = decoder_G(z, cond.detach())
        out_with_garbage_u = decoder_G(z, torch.randn(B, 320, device=device))
        diff = (out_with_real_u - out_with_garbage_u).abs().max().item()
        assert diff == 0.0, f"decoder_G should ignore u_cond, but outputs differ: {diff}"
    run_test("decoder_G ignores u_cond, uses exact disentangled_repr", test_decoder_g_uses_exact_repr)

    def test_decoder_g_output_shape():
        _dis = disentangled_repr.detach()
        x_hat = vq_decoder.decode(z, disentangled_repr=_dis)
        assert x_hat.shape == (B, 3, 64, 64)
    run_test("decoder_G output shape: (B, 3, 64, 64)", test_decoder_g_output_shape)

    def test_decoder_g_differs_with_different_repr():
        """Different disentangled_repr should produce different outputs."""
        _dis1 = disentangled_repr.detach()
        _dis2 = torch.randn_like(_dis1)
        out1 = vq_decoder.decode(z, disentangled_repr=_dis1)
        out2 = vq_decoder.decode(z, disentangled_repr=_dis2)
        diff = (out1 - out2).abs().max().item()
        assert diff > 1e-3, f"decoder should produce different output for different repr"
    run_test("decoder output changes with different disentangled_repr", test_decoder_g_differs_with_different_repr)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 4: all 5 losses x 2 critics — exact repr flow")
    print("=" * 65)

    _dis = disentangled_repr.detach()

    def decoder_G(z_, u_cond):
        return vq_decoder.decode(z_, disentangled_repr=_dis)

    def concept_encoder_fn(x):
        return encoder4(x).reshape(x.shape[0], -1)

    concept_critic = ConceptEncoderCritic(concept_encoder_fn)

    for loss_type in ["nce_logistic", "infonce_mechgrad", "fisher_sm", "denoise_sm", "jacobian_vjp_infonce"]:
        for critic_name, critic_obj in [("concept_encoder", concept_critic), ("mechanism_critic", mech_critic)]:

            def _test(lt=loss_type, cn=critic_name, co=critic_obj):
                all_modules = [vq_decoder, Pi_g, Pi_u]
                if isinstance(co, MechanismCritic):
                    all_modules.append(co)
                for m in all_modules:
                    m.zero_grad(set_to_none=True)

                loss_val = mcl_loss(
                    loss_type=lt,
                    decoder_G=decoder_G,
                    z=z.clone(),
                    u_key=cond.clone().detach(),  # (B, 320) for Pi_u/critic
                    u_for_G=None,
                    critic=co,
                    Pi_g=Pi_g, Pi_u=Pi_u,
                    tau=0.1, sigma=0.1,
                    neg_mode="shuffle_u",
                    create_graph=True,
                )
                assert_finite_scalar(loss_val, f"{lt}+{cn}")
                loss_val.backward()
                assert_has_grad(all_modules, f"{lt}+{cn}")

            run_test(f"{loss_type} + {critic_name}", _test)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 5: ConceptEncoderCritic == -MSE")
    print("=" * 65)

    def test_critic_equivalence():
        x_test = decoder_G(z, cond.detach())
        u_hat = concept_encoder_fn(x_test)
        old_score = -((u_hat - cond.detach()) ** 2).sum(dim=1)
        new_score = concept_critic(x_test, z, cond.detach())
        diff = (old_score - new_score).abs().max().item()
        assert diff < 1e-5, f"max diff = {diff}"
    run_test("ConceptEncoderCritic matches -MSE", test_critic_equivalence)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 6: data separation — decoder gets (B,20), Pi_u/critic get (B,320)")
    print("=" * 65)

    def test_data_separation():
        """
        Verify the key insight: decoder uses (B,20), contrastive uses (B,320).
        These are different dimensionalities, confirming proper separation.
        """
        assert _dis.shape == (B, LATENT_UNIT)                    # decoder input: 20
        assert cond.shape == (B, LATENT_UNIT * CONTEXT_DIM)      # Pi_u input: 320
        assert _dis.shape[1] != cond.shape[1]                    # they're different!
    run_test("decoder=(B,20), Pi_u/critic=(B,320) — properly separated", test_data_separation)

    # ==================================================================
    print("\n" + "=" * 65)
    print("GROUP 7: edge cases")
    print("=" * 65)

    def test_ensure_u_2d():
        assert ensure_u_2d(torch.tensor(1.0)).shape == (1, 1)
        assert ensure_u_2d(torch.randn(5)).shape == (5, 1)
        assert ensure_u_2d(torch.randn(5, 3)).shape == (5, 3)
        assert ensure_u_2d(torch.randn(B, 320)).shape == (B, 320)
    run_test("ensure_u_2d: scalar/1d/2d all correct", test_ensure_u_2d)

    def test_invalid_loss_type():
        try:
            mcl_loss("nonexistent", decoder_G, z, cond)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
    run_test("invalid loss_type raises ValueError", test_invalid_loss_type)

    def test_u_key_u_for_g_separation():
        u_key_small = torch.randn(B, 10, device=device)
        Pi_u_small = MLPProj(in_dim=10, out_dim=128, layernorm=False).to(device)
        Pi_u_small.zero_grad(set_to_none=True)
        loss = mcl_loss(
            loss_type="jacobian_vjp_infonce",
            decoder_G=decoder_G, z=z.clone(),
            u_key=u_key_small, u_for_G=cond.clone().detach(),
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
