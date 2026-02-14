"""
Test: checkpoint loading compatibility & gradient flow through frozen VQ-VAE.

Covers:
  GROUP 1 — Checkpoint loading compatibility:
    1. strict=False load: missing MCL keys (Pi_g, Pi_u, mcl_critic) are tolerated
    2. Missing keys are correctly reported
    3. Existing weights are loaded correctly (not randomly re-initialized)
    4. MCL modules initialize with fresh random weights after partial load
    5. No unexpected keys when loading a pre-MCL checkpoint

  GROUP 2 — Gradient flow through frozen VQ-VAE decoder:
    1. VQ-VAE decoder params have requires_grad=False
    2. Gradients still flow through decoder activations to z input
    3. Gradients flow through decoder activations to disentangled_repr
    4. Gradients do NOT accumulate on frozen VQ-VAE params
    5. Full MCL backward: encoder grad flows through frozen decoder
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ldm.models.diffusion.mcl_utils import (
    MLPProj, MechanismCritic, mcl_loss
)


# =========================================================================
# Dummy modules reproducing EXACT EncDiff shapes and interfaces
# =========================================================================

class DummyVQDecoder(nn.Module):
    """
    Mimics VQModelInterface.decode(h, disentangled_repr=u):
      h: (B, 3, 16, 16), disentangled_repr: (B, 20) -> output: (B, 3, 64, 64)
    Includes use_disentangled_concat flag and post_quant_conv like real model.
    """
    def __init__(self, embed_dim=3, disentangled_dim=20):
        super().__init__()
        self.use_disentangled_concat = True
        self.disentangled_dim = disentangled_dim
        self.post_quant_conv = nn.Conv2d(embed_dim + disentangled_dim, embed_dim, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 3, 3, padding=1),
        )

    def decode(self, h, force_not_quantize=False, disentangled_repr=None):
        B, _, H, W = h.shape
        if self.use_disentangled_concat:
            if disentangled_repr is not None:
                s_expanded = disentangled_repr[:, :, None, None].expand(-1, -1, H, W)
            else:
                s_expanded = torch.zeros(B, self.disentangled_dim, H, W,
                                         device=h.device, dtype=h.dtype)
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


class DummyLatentDiffusion(nn.Module):
    """
    Minimal reproduction of LatentDiffusion's checkpoint-related structure.
    Contains: model (UNet-like), first_stage_model (VQ-VAE), cond_stage_model (Encoder4),
    and optionally Pi_g, Pi_u, mcl_critic (MCL modules).
    """
    def __init__(self, add_mcl=False):
        super().__init__()
        # Core modules (present in pre-MCL checkpoint)
        self.model = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.first_stage_model = DummyVQDecoder()
        self.cond_stage_model = DummyEncoder4()
        self.logvar = nn.Parameter(torch.zeros(1000))
        self.scale_factor = nn.Parameter(torch.ones(1), requires_grad=False)

        # MCL modules (NOT in pre-MCL checkpoint)
        if add_mcl:
            self.Pi_g = MLPProj(in_dim=3*16*16, out_dim=128, layernorm=True)
            self.Pi_u = MLPProj(in_dim=20, out_dim=128, layernorm=False)
            self.mcl_critic = MechanismCritic(z_shape=(3, 16, 16), u_dim=20, hidden=256)


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


# =========================================================================
# GROUP 1: Checkpoint loading compatibility
# =========================================================================

def run_ckpt_tests():
    global passed, failed

    print("=" * 65)
    print("GROUP 1: Checkpoint loading compatibility (init_from_ckpt)")
    print("=" * 65)

    torch.manual_seed(42)

    # Step 1: Create a "pre-MCL" model and save its state_dict
    # This simulates checkpoint 2 (concat EncDiff, no MCL)
    pre_mcl_model = DummyLatentDiffusion(add_mcl=False)
    pre_mcl_sd = pre_mcl_model.state_dict()
    pre_mcl_keys = set(pre_mcl_sd.keys())

    # Step 2: Create an "MCL" model (with Pi_g, Pi_u, mcl_critic)
    torch.manual_seed(99)  # different seed so random init differs
    mcl_model = DummyLatentDiffusion(add_mcl=True)
    mcl_keys = set(mcl_model.state_dict().keys())

    # Identify MCL-only keys
    mcl_only_keys = mcl_keys - pre_mcl_keys
    print(f"  [info] MCL-only keys ({len(mcl_only_keys)}): "
          f"{sorted([k.split('.')[0] for k in mcl_only_keys])[:5]}...")

    # ── Test 1: strict=False load succeeds ──
    def test_strict_false_load():
        missing, unexpected = mcl_model.load_state_dict(pre_mcl_sd, strict=False)
        assert len(missing) > 0, "Should have missing MCL keys"
        assert len(unexpected) == 0, f"Should have no unexpected keys, got: {unexpected}"
    run_test("strict=False load succeeds (missing MCL keys tolerated)", test_strict_false_load)

    # ── Test 2: Missing keys are exactly the MCL modules ──
    def test_missing_keys_are_mcl():
        missing, _ = mcl_model.load_state_dict(pre_mcl_sd, strict=False)
        missing_set = set(missing)
        # Every missing key should start with Pi_g, Pi_u, or mcl_critic
        mcl_prefixes = ("Pi_g.", "Pi_u.", "mcl_critic.")
        for k in missing_set:
            assert any(k.startswith(p) for p in mcl_prefixes), \
                f"Unexpected missing key: {k} (expected only MCL module keys)"
        # All MCL-only keys should be in missing
        assert mcl_only_keys == missing_set, \
            f"Missing keys mismatch: extra={missing_set - mcl_only_keys}, absent={mcl_only_keys - missing_set}"
    run_test("missing keys are exactly Pi_g, Pi_u, mcl_critic params", test_missing_keys_are_mcl)

    # ── Test 3: Existing weights loaded correctly ──
    def test_existing_weights_loaded():
        # Re-load from pre_mcl_sd
        mcl_model2 = DummyLatentDiffusion(add_mcl=True)
        mcl_model2.load_state_dict(pre_mcl_sd, strict=False)
        # Check that shared keys match exactly
        for key in pre_mcl_keys:
            loaded = mcl_model2.state_dict()[key]
            original = pre_mcl_sd[key]
            diff = (loaded - original).abs().max().item()
            assert diff == 0.0, f"Key {key} not loaded correctly, max diff = {diff}"
    run_test("existing weights (model, first_stage, cond_stage) loaded exactly", test_existing_weights_loaded)

    # ── Test 4: MCL modules have fresh random weights (not zeros or pre_mcl weights) ──
    def test_mcl_modules_random_init():
        torch.manual_seed(99)
        fresh_model = DummyLatentDiffusion(add_mcl=True)
        # Load pre-MCL checkpoint
        fresh_model.load_state_dict(pre_mcl_sd, strict=False)
        # MCL weights should still be their initial random values (from seed 99)
        # They should NOT be all-zeros
        for name in ["Pi_g", "Pi_u", "mcl_critic"]:
            module = getattr(fresh_model, name)
            total_abs = sum(p.abs().sum().item() for p in module.parameters())
            assert total_abs > 0, f"{name} params are all-zero after partial load"
    run_test("MCL modules retain random init after partial ckpt load", test_mcl_modules_random_init)

    # ── Test 5: No unexpected keys when loading pre-MCL ckpt ──
    def test_no_unexpected_keys():
        mcl_model3 = DummyLatentDiffusion(add_mcl=True)
        _, unexpected = mcl_model3.load_state_dict(pre_mcl_sd, strict=False)
        assert len(unexpected) == 0, f"Got {len(unexpected)} unexpected keys: {unexpected[:5]}"
    run_test("no unexpected keys from pre-MCL checkpoint", test_no_unexpected_keys)

    # ── Test 6: strict=True WOULD fail (sanity check) ──
    def test_strict_true_fails():
        mcl_model4 = DummyLatentDiffusion(add_mcl=True)
        try:
            mcl_model4.load_state_dict(pre_mcl_sd, strict=True)
            raise AssertionError("strict=True should raise RuntimeError for missing keys")
        except RuntimeError:
            pass  # Expected
    run_test("strict=True correctly rejects partial checkpoint", test_strict_true_fails)

    # ── Test 7: Simulate full init_from_ckpt flow ──
    def test_full_init_from_ckpt_flow():
        """Simulate the exact init_from_ckpt logic from ddpm_enc.py."""
        # Save as a full checkpoint dict (like PyTorch Lightning)
        import tempfile
        ckpt = {"state_dict": pre_mcl_sd, "epoch": 100, "global_step": 50000}
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            torch.save(ckpt, f.name)
            ckpt_path = f.name

        try:
            mcl_model5 = DummyLatentDiffusion(add_mcl=True)
            # Reproduce init_from_ckpt logic
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            ignore_keys = []
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        del sd[k]
            missing, unexpected = mcl_model5.load_state_dict(sd, strict=False)
            assert len(missing) > 0, "Should have missing MCL keys"
            assert len(unexpected) == 0, "Should have no unexpected keys"
            # Model should be functional
            with torch.no_grad():
                dummy_z = torch.randn(2, 3, 16, 16)
                dummy_repr = torch.randn(2, 20)
                out = mcl_model5.first_stage_model.decode(dummy_z, disentangled_repr=dummy_repr)
                assert out.shape == (2, 3, 64, 64), f"Unexpected output shape: {out.shape}"
        finally:
            os.unlink(ckpt_path)
    run_test("full init_from_ckpt flow: load + decode works", test_full_init_from_ckpt_flow)


# =========================================================================
# GROUP 2: Gradient flow through frozen VQ-VAE decoder
# =========================================================================

def run_gradient_tests():
    global passed, failed

    print("\n" + "=" * 65)
    print("GROUP 2: Gradient flow through frozen VQ-VAE decoder")
    print("=" * 65)

    torch.manual_seed(42)
    B = 4
    LATENT_UNIT = 20

    # Create a frozen VQ-VAE decoder (mimics instantiate_first_stage)
    vq_decoder = DummyVQDecoder(embed_dim=3, disentangled_dim=LATENT_UNIT)
    vq_decoder.eval()
    for param in vq_decoder.parameters():
        param.requires_grad = False

    # ── Test 1: VQ-VAE params are frozen ──
    def test_vq_params_frozen():
        for name, param in vq_decoder.named_parameters():
            assert not param.requires_grad, f"VQ-VAE param {name} has requires_grad=True"
    run_test("all VQ-VAE decoder params have requires_grad=False", test_vq_params_frozen)

    # ── Test 2: Gradients flow through decoder to z input ──
    def test_grad_flows_to_z():
        z = torch.randn(B, 3, 16, 16, requires_grad=True)
        dis_repr = torch.randn(B, LATENT_UNIT)  # no grad needed for this test
        out = vq_decoder.decode(z, disentangled_repr=dis_repr)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None, "z.grad is None — gradients not flowing through frozen decoder"
        assert z.grad.abs().sum() > 0, "z.grad is all-zero"
    run_test("gradients flow through frozen decoder to z input", test_grad_flows_to_z)

    # ── Test 3: Gradients flow through decoder to disentangled_repr ──
    def test_grad_flows_to_repr():
        z = torch.randn(B, 3, 16, 16)
        dis_repr = torch.randn(B, LATENT_UNIT, requires_grad=True)
        out = vq_decoder.decode(z, disentangled_repr=dis_repr)
        loss = out.sum()
        loss.backward()
        assert dis_repr.grad is not None, "disentangled_repr.grad is None"
        assert dis_repr.grad.abs().sum() > 0, "disentangled_repr.grad is all-zero"
    run_test("gradients flow through frozen decoder to disentangled_repr", test_grad_flows_to_repr)

    # ── Test 4: Frozen VQ-VAE params do NOT accumulate gradients ──
    def test_frozen_params_no_grad():
        z = torch.randn(B, 3, 16, 16, requires_grad=True)
        dis_repr = torch.randn(B, LATENT_UNIT, requires_grad=True)
        out = vq_decoder.decode(z, disentangled_repr=dis_repr)
        loss = out.sum()
        loss.backward()
        for name, param in vq_decoder.named_parameters():
            assert param.grad is None, \
                f"Frozen param {name} has gradient (should be None)"
    run_test("frozen VQ-VAE params do NOT accumulate gradients", test_frozen_params_no_grad)

    # ── Test 5: Simultaneous grad flow to both z and repr ──
    def test_simultaneous_grad_flow():
        z = torch.randn(B, 3, 16, 16, requires_grad=True)
        dis_repr = torch.randn(B, LATENT_UNIT, requires_grad=True)
        out = vq_decoder.decode(z, disentangled_repr=dis_repr)
        loss = out.mean()
        loss.backward()
        assert z.grad is not None and z.grad.abs().sum() > 0
        assert dis_repr.grad is not None and dis_repr.grad.abs().sum() > 0
    run_test("simultaneous gradient flow to z and disentangled_repr", test_simultaneous_grad_flow)

    # ── Test 6: Full MCL pipeline — Pi_g/Pi_u/critic get grads through frozen decoder ──
    def test_mcl_full_pipeline_gradient():
        """
        End-to-end: MCL loss -> backward through frozen VQ-VAE.
        Uses MechanismCritic (senior's design), all components use u (B, 20).
        """
        Pi_g = MLPProj(in_dim=3*16*16, out_dim=128, layernorm=True)
        Pi_u = MLPProj(in_dim=LATENT_UNIT, out_dim=128, layernorm=False)
        critic = MechanismCritic(z_shape=(3, 16, 16), u_dim=LATENT_UNIT, hidden=256)

        u_mcl = torch.randn(B, LATENT_UNIT)  # (B, 20)
        z = torch.randn(B, 3, 16, 16)

        def decoder_G(z_, u_cond):
            return vq_decoder.decode(z_, disentangled_repr=u_cond)

        # Zero grads
        Pi_g.zero_grad(set_to_none=True)
        Pi_u.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)

        loss_val = mcl_loss(
            loss_type="infonce_mechgrad",
            decoder_G=decoder_G,
            z=z,
            u_key=u_mcl,  # (B, 20) for everything
            u_for_G=None,
            critic=critic,
            Pi_g=Pi_g, Pi_u=Pi_u,
            tau=0.1, sigma=0.1,
            neg_mode="shuffle_u",
            create_graph=True,  # same as real training (ddpm_enc.py p_losses)
        )
        assert torch.isfinite(loss_val), f"MCL loss not finite: {loss_val.item()}"
        loss_val.backward()

        # Check Pi_g, Pi_u, critic received gradients
        for name, module in [("Pi_g", Pi_g), ("Pi_u", Pi_u), ("critic", critic)]:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.parameters()
            )
            assert has_grad, f"{name} has no gradients"

        # Check VQ-VAE is still frozen (no param grads)
        for name, param in vq_decoder.named_parameters():
            assert param.grad is None, \
                f"Frozen VQ-VAE param {name} received gradient during MCL backward"

    run_test("full MCL pipeline: Pi_g/Pi_u/critic grads flow, VQ-VAE stays frozen", test_mcl_full_pipeline_gradient)

    # ── Test 6b: Encoder gets grads through frozen decoder via direct loss ──
    def test_encoder_grad_through_frozen_decoder():
        """
        Verify encoder4 CAN receive gradients through frozen VQ-VAE
        when the loss directly depends on the decoder output and
        disentangled_repr is NOT detached.

        This is the scenario for diffusion loss: encoder produces repr,
        decoder uses it, loss backprops through decoder to encoder.
        """
        encoder4_local = DummyEncoder4(latent_unit=LATENT_UNIT, context_dim=16)
        encoder4_local.zero_grad(set_to_none=True)

        dummy_img = torch.randn(B, 3, 64, 64)
        dis_repr = encoder4_local.encoding(dummy_img)  # (B, 20), graph attached

        z = torch.randn(B, 3, 16, 16)
        x_hat = vq_decoder.decode(z, disentangled_repr=dis_repr)  # through frozen decoder

        # Direct loss on output (simulates reconstruction loss)
        target = torch.randn_like(x_hat)
        loss = F.mse_loss(x_hat, target)
        loss.backward()

        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder4_local.parameters()
        )
        assert encoder_has_grad, \
            "encoder4 should get gradients through frozen VQ-VAE when repr is not detached"

        # VQ-VAE still frozen
        for name, param in vq_decoder.named_parameters():
            assert param.grad is None, \
                f"Frozen VQ-VAE param {name} received gradient"

    run_test("encoder gets grads through frozen decoder via direct loss on output",
             test_encoder_grad_through_frozen_decoder)

    # ── Test 7: differentiable_decode_first_stage simulation ──
    def test_differentiable_decode_simulation():
        """
        Simulate the exact differentiable_decode_first_stage path:
        z -> scale -> VQ decode -> output.
        Verify gradient flows despite frozen decoder.
        """
        scale_factor = 0.5  # typical value
        z = torch.randn(B, 3, 16, 16, requires_grad=True)
        dis_repr = torch.randn(B, LATENT_UNIT, requires_grad=True)

        # Reproduce differentiable_decode_first_stage
        z_scaled = 1.0 / scale_factor * z
        out = vq_decoder.decode(z_scaled, disentangled_repr=dis_repr)

        loss = out.mean()
        loss.backward()

        assert z.grad is not None and z.grad.abs().sum() > 0, \
            "z.grad missing after differentiable_decode path"
        assert dis_repr.grad is not None and dis_repr.grad.abs().sum() > 0, \
            "disentangled_repr.grad missing after differentiable_decode path"
    run_test("differentiable_decode_first_stage: grads flow through scale + frozen decode",
             test_differentiable_decode_simulation)

    # ── Test 8: @torch.no_grad vs no decorator difference ──
    def test_no_grad_blocks_gradient():
        """
        Verify that @torch.no_grad() WOULD block gradients.
        This confirms why differentiable_decode_first_stage (no decorator) is correct,
        while decode_first_stage (@torch.no_grad) would NOT work for MCL.
        """
        z = torch.randn(B, 3, 16, 16, requires_grad=True)
        dis_repr = torch.randn(B, LATENT_UNIT, requires_grad=True)

        # With @torch.no_grad — gradients should NOT flow
        with torch.no_grad():
            out_ngrad = vq_decoder.decode(z, disentangled_repr=dis_repr)
        # out_ngrad has no grad_fn
        assert not out_ngrad.requires_grad, \
            "Output under torch.no_grad should not require grad"

        # Without @torch.no_grad — gradients SHOULD flow
        out_grad = vq_decoder.decode(z, disentangled_repr=dis_repr)
        assert out_grad.requires_grad, \
            "Output without torch.no_grad should require grad (from z and dis_repr)"
    run_test("torch.no_grad blocks grads; no decorator allows them (MCL needs latter)",
             test_no_grad_blocks_gradient)


# =========================================================================
# Main
# =========================================================================

def run_all():
    run_ckpt_tests()
    run_gradient_tests()

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
