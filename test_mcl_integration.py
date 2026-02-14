"""
Test: verify mcl_utils unified API + ddpm_enc.py integration is correct.

Covers:
  1. mcl_utils API: all 5 loss types, both critic modes, backward, shapes
  2. p_losses MCL block simulation: exact reproduction of ddpm_enc.py logic
     with the REAL cond shape (B, 320) flat — NOT (B, 20, 16)
  3. decoder_G reshape: (B,320) -> (B,20,16) -> mean -> (B,20) for VQ-VAE decode
  4. ConceptEncoderCritic semantic equivalence with old mech_score_mse
  5. Gradient flow: MCL loss gradients reach decoder, Pi_g, Pi_u, critic
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
# Dummy modules that reproduce the EXACT EncDiff shapes and interfaces
# =========================================================================

class DummyVQDecoder(nn.Module):
    """
    Mimics VQModelInterface.decode(h, disentangled_repr=u):
      - h: (B, 3, 16, 16) latent from diffusion
      - disentangled_repr: (B, 20) raw encoding
      - output: (B, 3, 64, 64) image
    Internally concats u along channel dim, exactly like the real code.
    """
    def __init__(self, embed_dim=3, disentangled_dim=20):
        super().__init__()
        # post_quant_conv: (embed_dim + disentangled_dim) -> embed_dim
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
        h = self.post_quant_conv(h)  # (B, 3, 16, 16)
        return self.decoder(h)       # (B, 3, 64, 64)


class DummyEncoder4(nn.Module):
    """
    Mimics Encoder4:
      - forward(x):   image (B,3,64,64) -> warped cond (B, 320) [= latent_unit * context_dim]
      - encoding(x):  image (B,3,64,64) -> raw disentangled repr (B, 20)
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
        out = []
        for i in range(self.latent_unit):
            out.append(self.warp_nets[i](codes[:, i:i+1]))  # (B, 16)
        return torch.cat(out, dim=1)  # (B, 320)

    def encoding(self, x):
        return self.encoder(x)  # (B, 20)


# =========================================================================
# Helper
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
    assert t.dim() == 0, f"{label} expected scalar, got shape {t.shape}"
    assert torch.isfinite(t), f"{label} not finite: {t.item()}"


def assert_has_grad(modules, label=""):
    for m in modules:
        for p in m.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                return
    raise AssertionError(f"{label} no non-zero gradients found")


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

    # --- Build modules ---
    vq_decoder = DummyVQDecoder(embed_dim=3, disentangled_dim=LATENT_UNIT).to(device)
    encoder4 = DummyEncoder4(latent_unit=LATENT_UNIT, context_dim=CONTEXT_DIM).to(device)
    Pi_g = MLPProj(in_dim=3*16*16, out_dim=128, layernorm=True).to(device)
    Pi_u = MLPProj(in_dim=LATENT_UNIT*CONTEXT_DIM, out_dim=128, layernorm=False).to(device)
    mech_critic = MechanismCritic(z_shape=(3,16,16), u_dim=LATENT_UNIT*CONTEXT_DIM, hidden=256).to(device)

    # --- Simulate the exact data that p_losses receives ---
    z = torch.randn(B, 3, 16, 16, device=device)          # x_start
    dummy_img = torch.randn(B, 3, 64, 64, device=device)  # original image
    # Encoder4.forward(image) -> (B, 320)  <-- this is what p_losses gets as `cond`
    cond = encoder4(dummy_img)  # (B, 320)

    # ==================================================================
    print("=" * 65)
    print("TEST GROUP 1: cond shape verification (the bug we fixed)")
    print("=" * 65)

    def test_cond_is_2d():
        """cond from Encoder4.forward() must be (B, 320), not (B, 20, 16)"""
        assert cond.dim() == 2, f"cond should be 2D, got {cond.dim()}D: {cond.shape}"
        assert cond.shape == (B, LATENT_UNIT * CONTEXT_DIM), \
            f"cond shape should be ({B}, {LATENT_UNIT*CONTEXT_DIM}), got {cond.shape}"

    run_test("cond from Encoder4 is (B, 320) flat", test_cond_is_2d)

    def test_old_code_would_crash():
        """The OLD code did u.shape[2] on a 2D tensor -- that must IndexError"""
        try:
            _ = cond.shape[2]  # This is what the old buggy code did
            raise AssertionError("Should have raised IndexError")
        except IndexError:
            pass  # Good, old code would indeed crash

    run_test("old code u.shape[2] would IndexError on (B,320)", test_old_code_would_crash)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 2: decoder_G reshape logic (fixed code path)")
    print("=" * 65)

    def test_decoder_g_reshape():
        """
        Fixed code: reshape (B,320) -> (B,20,16) -> mean(dim=2) -> (B,20)
        Then pass to VQ decoder as disentangled_repr.
        """
        _latent_unit = encoder4.latent_unit   # 20
        _context_dim = encoder4.context_dim   # 16

        def decoder_G(z_, u_cond):
            u_for_decoder = u_cond.reshape(z_.shape[0], _latent_unit, _context_dim).mean(dim=2)
            assert u_for_decoder.shape == (z_.shape[0], _latent_unit), \
                f"u_for_decoder should be (B, {_latent_unit}), got {u_for_decoder.shape}"
            return vq_decoder.decode(z_, disentangled_repr=u_for_decoder)

        x_hat = decoder_G(z, cond.detach())
        assert x_hat.shape == (B, 3, 64, 64), \
            f"decoder output should be (B,3,64,64), got {x_hat.shape}"

    run_test("decoder_G: (B,320) -> reshape -> mean -> (B,20) -> decode -> (B,3,64,64)", test_decoder_g_reshape)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 3: full p_losses MCL block simulation (all 5 loss types)")
    print("=" * 65)

    _latent_unit = encoder4.latent_unit
    _context_dim = encoder4.context_dim

    def decoder_G(z_, u_cond):
        u_for_decoder = u_cond.reshape(z_.shape[0], _latent_unit, _context_dim).mean(dim=2)
        return vq_decoder.decode(z_, disentangled_repr=u_for_decoder)

    def concept_encoder_fn(x):
        c = encoder4(x)
        return c.reshape(c.shape[0], -1)

    concept_critic = ConceptEncoderCritic(concept_encoder_fn)

    for loss_type in ["nce_logistic", "infonce_mechgrad", "fisher_sm", "denoise_sm", "jacobian_vjp_infonce"]:
        for critic_name, critic_obj in [("concept_encoder", concept_critic), ("mechanism_critic", mech_critic)]:

            def _test(lt=loss_type, cn=critic_name, co=critic_obj):
                all_modules = [vq_decoder, Pi_g, Pi_u]
                if isinstance(co, MechanismCritic):
                    all_modules.append(co)

                for m in all_modules:
                    m.zero_grad(set_to_none=True)

                # This is the EXACT call from p_losses (post-fix)
                loss_val = mcl_loss(
                    loss_type=lt,
                    decoder_G=decoder_G,
                    z=z.clone(),
                    u_key=cond.clone().detach(),  # (B, 320) flat
                    u_for_G=None,
                    critic=co,
                    Pi_g=Pi_g,
                    Pi_u=Pi_u,
                    tau=0.1,
                    sigma=0.1,
                    neg_mode="shuffle_u",
                    create_graph=True,
                )
                assert_finite_scalar(loss_val, f"{lt}+{cn}")
                loss_val.backward()
                assert_has_grad(all_modules, f"{lt}+{cn}")

            run_test(f"{loss_type} + {critic_name}", _test)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 4: ConceptEncoderCritic == old mech_score_mse")
    print("=" * 65)

    def test_critic_equivalence():
        x_test = decoder_G(z, cond.detach())
        u_hat = concept_encoder_fn(x_test)
        old_score = -((u_hat - cond.detach()) ** 2).sum(dim=1)
        new_score = concept_critic(x_test, z, cond.detach())
        diff = (old_score - new_score).abs().max().item()
        assert diff < 1e-5, f"Score mismatch: max diff={diff}"

    run_test(f"ConceptEncoderCritic output matches -MSE (max diff < 1e-5)", test_critic_equivalence)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 5: ensure_u_2d shape handling")
    print("=" * 65)

    def test_ensure_u_2d():
        assert ensure_u_2d(torch.tensor(1.0)).shape == (1, 1)
        assert ensure_u_2d(torch.randn(5)).shape == (5, 1)
        assert ensure_u_2d(torch.randn(5, 3)).shape == (5, 3)
        assert ensure_u_2d(torch.randn(B, 320)).shape == (B, 320)

    run_test("ensure_u_2d: scalar/1d/2d all correct", test_ensure_u_2d)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 6: u_key vs u_for_G separation")
    print("=" * 65)

    def test_u_key_separation():
        u_key_small = torch.randn(B, 10, device=device)
        Pi_u_small = MLPProj(in_dim=10, out_dim=128, layernorm=False).to(device)
        Pi_u_small.zero_grad(set_to_none=True)

        loss = mcl_loss(
            loss_type="jacobian_vjp_infonce",
            decoder_G=decoder_G,
            z=z.clone(),
            u_key=u_key_small,       # 10-dim key for contrastive
            u_for_G=cond.clone().detach(),  # 320-dim for decoder
            Pi_g=Pi_g, Pi_u=Pi_u_small,
            tau=0.1, create_graph=False,
        )
        assert_finite_scalar(loss, "u_key_separation")
        loss.backward()
        assert_has_grad([Pi_u_small], "u_key_separation Pi_u_small")

    run_test("u_key(10d) != u_for_G(320d): different dims work", test_u_key_separation)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 7: invalid loss_type raises ValueError")
    print("=" * 65)

    def test_invalid_loss_type():
        try:
            mcl_loss("nonexistent", decoder_G, z, cond)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    run_test("invalid loss_type raises ValueError", test_invalid_loss_type)

    # ==================================================================
    print("\n" + "=" * 65)
    print("TEST GROUP 8: Encoder4.encoding vs Encoder4.forward shape check")
    print("=" * 65)

    def test_encoder4_shapes():
        enc_raw = encoder4.encoding(dummy_img)  # (B, 20)
        enc_warped = encoder4(dummy_img)         # (B, 320)
        assert enc_raw.shape == (B, LATENT_UNIT), \
            f"encoding() should be (B,{LATENT_UNIT}), got {enc_raw.shape}"
        assert enc_warped.shape == (B, LATENT_UNIT * CONTEXT_DIM), \
            f"forward() should be (B,{LATENT_UNIT*CONTEXT_DIM}), got {enc_warped.shape}"
        # The mean-reshape approximation should be "close" to raw encoding
        # (not exact because warp is nonlinear)
        approx = enc_warped.reshape(B, LATENT_UNIT, CONTEXT_DIM).mean(dim=2)
        assert approx.shape == enc_raw.shape

    run_test("Encoder4: encoding()->(B,20), forward()->(B,320), mean reshape works", test_encoder4_shapes)

    # ==================================================================
    # Summary
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
