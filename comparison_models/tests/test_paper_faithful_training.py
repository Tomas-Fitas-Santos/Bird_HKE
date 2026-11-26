"""CPU-only checks for the paper-faithful optimizer and LR schedule.

ViTPose-B and PoseBH-B both train with a layer-decayed AdamW and an
iteration-based warmup + step schedule. These tests pin the values from the
pinned upstream configs and the behaviour of the local reimplementation.
"""
import unittest

try:
    import torch
    from comparison_models.common.training_utils import (
        PaperScheduleLR, layer_decay_param_groups, make_optimizer,
        make_scheduler,
    )
    from comparison_models.posebh.recipe import RECIPE as POSEBH_RECIPE
    from comparison_models.vitpose.recipe import RECIPE as VIPOISE_RECIPE
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


if HAS_DEPS:
    class _Attention(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.qkv = torch.nn.Linear(dim, dim * 3)
            self.proj = torch.nn.Linear(dim, dim)

    class _Mlp(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = torch.nn.Linear(dim, dim * 2)
            self.fc2 = torch.nn.Linear(dim * 2, dim)

    class _Block(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = _Attention(dim)
            self.norm1 = torch.nn.LayerNorm(dim)
            self.mlp = _Mlp(dim)
            self.norm2 = torch.nn.LayerNorm(dim)

    class _PatchEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 4, 2, stride=2)

    class _TinyViT(torch.nn.Module):
        """Uses the same parameter naming as ViTPoseB / PoseBHBirdGazeB."""

        def __init__(self, depth=3):
            super().__init__()
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, 5, 4))
            self.patch_embed = _PatchEmbed()
            self.blocks = torch.nn.ModuleList(_Block(4) for _ in range(depth))
            self.last_norm = torch.nn.LayerNorm(4)
            self.final_layer = torch.nn.Conv2d(4, 4, 3, padding=1)
            self.heatmap_head = torch.nn.Conv2d(4, 4, 1)


@unittest.skipUnless(HAS_DEPS, "Requires torch")
class LayerDecayParamGroupTests(unittest.TestCase):
    def groups(self, model=None, **kwargs):
        options = dict(base_lr=1.0, weight_decay=0.1, num_layers=3,
                       layer_decay_rate=0.5)
        options.update(kwargs)
        return layer_decay_param_groups(model or _TinyViT(), **options)

    def test_layer_groups_and_lr_scales_follow_upstream_constructor(self):
        # num_layers becomes 3 + 2 = 5, so layers 0..4 exist, each split into
        # a decay and a no_decay group.
        by_name = {group["group_name"]: group for group in self.groups()}
        self.assertEqual(
            sorted(by_name),
            ["layer_%d_%s" % (layer, kind)
             for layer in range(5) for kind in ("decay", "no_decay")])
        # scale = layer_decay_rate ** (num_layers - layer_id - 1)
        self.assertAlmostEqual(by_name["layer_0_decay"]["lr_scale"], 0.5 ** 4)
        self.assertAlmostEqual(by_name["layer_3_decay"]["lr_scale"], 0.5 ** 1)
        self.assertAlmostEqual(by_name["layer_4_decay"]["lr_scale"], 1.0)
        self.assertAlmostEqual(by_name["layer_4_decay"]["lr"], 1.0)
        self.assertAlmostEqual(by_name["layer_0_decay"]["lr"], 0.5 ** 4)

    def test_patch_embed_is_layer_zero_and_heads_are_last_layer(self):
        layers = {}
        for group in self.groups():
            for name in group["param_names"]:
                layers[name] = int(group["group_name"].split("_")[1])
        self.assertEqual(layers["pos_embed"], 0)
        self.assertEqual(layers["patch_embed.proj.weight"], 0)
        self.assertEqual(layers["blocks.0.attn.qkv.weight"], 1)
        self.assertEqual(layers["blocks.2.attn.qkv.weight"], 3)
        # Everything outside the transformer body shares the top layer.
        self.assertEqual(layers["final_layer.weight"], 4)
        self.assertEqual(layers["heatmap_head.weight"], 4)
        self.assertEqual(layers["last_norm.weight"], 4)

    def test_norms_biases_and_position_embedding_are_not_decayed(self):
        decayed = {}
        for group in self.groups():
            for name in group["param_names"]:
                decayed[name] = group["weight_decay"]
        self.assertEqual(decayed["pos_embed"], 0.0)
        self.assertEqual(decayed["patch_embed.proj.bias"], 0.0)
        self.assertEqual(decayed["blocks.1.norm1.weight"], 0.0)
        self.assertEqual(decayed["blocks.1.attn.qkv.bias"], 0.0)
        self.assertEqual(decayed["blocks.1.attn.qkv.weight"], 0.1)
        self.assertEqual(decayed["patch_embed.proj.weight"], 0.1)

    def test_every_parameter_lands_in_exactly_one_group(self):
        model = _TinyViT()
        names = [name for group in self.groups(model)
                 for name in group["param_names"]]
        self.assertEqual(sorted(names), sorted(dict(model.named_parameters())))

    def test_frozen_parameters_are_excluded(self):
        model = _TinyViT()
        for parameter in model.blocks[0].parameters():
            parameter.requires_grad_(False)
        names = [name for group in self.groups(model)
                 for name in group["param_names"]]
        self.assertFalse([name for name in names
                          if name.startswith("blocks.0.")])

    def test_optimizer_keeps_recipe_defaults_and_per_group_scales(self):
        model = _TinyViT()
        recipe = {"lr": 5e-4, "weight_decay": 0.1, "betas": (0.9, 0.999),
                  "schedule": {"type": "vit_layer_decay", "num_layers": 3,
                               "layer_decay_rate": 0.5}}
        optimizer = make_optimizer(recipe, model)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.1)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertAlmostEqual(
            max(group["lr_scale"] for group in optimizer.param_groups), 1.0)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"],
                                   group["lr_scale"] * 5e-4)
        # A recipe without a schedule block must keep the flat behaviour.
        flat = make_optimizer({"lr": 5e-4, "weight_decay": 0.01}, model)
        self.assertEqual(len(flat.param_groups), 1)


@unittest.skipUnless(HAS_DEPS, "Requires torch")
class PaperScheduleTests(unittest.TestCase):
    def scheduler(self, **kwargs):
        optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        options = dict(warmup_iters=4, warmup_ratio=0.001,
                       milestones_epochs=[5, 9], iters_per_epoch=10)
        options.update(kwargs)
        instance = PaperScheduleLR(optimizer, **options)
        # torch warns when a scheduler is stepped before its optimizer, and
        # these tests drive the schedule without a real parameter update.
        instance.optimizer.step()
        return instance

    def test_warmup_starts_at_ratio_and_reaches_one(self):
        scheduler = self.scheduler()
        optimizer = scheduler.optimizer
        factors = []
        for _ in range(6):
            factors.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        self.assertAlmostEqual(factors[0], 0.001)
        self.assertAlmostEqual(factors[3], 0.001 + 0.999 * 3 / 4)
        self.assertAlmostEqual(factors[4], 1.0)
        self.assertAlmostEqual(factors[5], 1.0)

    def test_step_decay_happens_at_epoch_milestones(self):
        scheduler = self.scheduler()
        optimizer = scheduler.optimizer
        seen = {}
        for iteration in range(105):
            if iteration in (0, 49, 50, 89, 90, 104):
                seen[iteration] = optimizer.param_groups[0]["lr"]
            scheduler.step()
        self.assertAlmostEqual(seen[0], 0.001)
        self.assertAlmostEqual(seen[49], 1.0)
        self.assertAlmostEqual(seen[50], 0.1)
        self.assertAlmostEqual(seen[89], 0.1)
        self.assertAlmostEqual(seen[90], 0.01)
        self.assertAlmostEqual(seen[104], 0.01)

    def test_scale_is_applied_per_group_on_top_of_layer_decay(self):
        model = _TinyViT()
        recipe = {"lr": 1.0, "weight_decay": 0.1, "betas": (0.9, 0.999),
                  "schedule": {"type": "vit_layer_decay", "num_layers": 3,
                               "layer_decay_rate": 0.5},
                  "lr_schedule": {"type": "vit_step", "warmup_iters": 2,
                                  "warmup_ratio": 0.5,
                                  "milestones_epochs": [1], "gamma": 0.1}}
        optimizer = make_optimizer(recipe, model)
        scheduler = make_scheduler(recipe, optimizer, iters_per_epoch=10)
        optimizer.step()
        before = [group["lr"] for group in optimizer.param_groups]
        for _ in range(12):
            scheduler.step()
        after = [group["lr"] for group in optimizer.param_groups]
        self.assertTrue(all(b < a for a, b in zip(before, after)))
        ratios = [b / a for a, b in zip(before, after)]
        self.assertTrue(all(abs(ratio - ratios[0]) < 1e-9 for ratio in ratios))

    def test_iteration_counter_survives_a_state_round_trip(self):
        scheduler = self.scheduler()
        for _ in range(7):
            scheduler.step()
        self.assertEqual(scheduler.iterations, 7)
        clone = self.scheduler()
        clone.load_state_dict(scheduler.state_dict())
        self.assertEqual(clone.iterations, 7)
        # Like every PyTorch scheduler, load_state_dict restores the counter
        # and the learning rate is recomputed by the next step.
        clone.step()
        self.assertAlmostEqual(clone.optimizer.param_groups[0]["lr"], 1.0)
        # A milestone that already passed must still be honoured after resume.
        passed = self.scheduler(milestones_epochs=[0])
        for _ in range(8):
            passed.step()
        clone.load_state_dict(passed.state_dict())
        clone.step()
        self.assertAlmostEqual(clone.optimizer.param_groups[0]["lr"], 0.1)

    def test_epoch_cosine_is_still_available_for_smoke_recipes(self):
        optimizer = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = make_scheduler(
            {"lr": 1.0, "epochs": 10, "warmup_epochs": 5, "min_lr": 0.02},
            optimizer, iters_per_epoch=10)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        self.assertFalse(getattr(scheduler, "steps_per_iteration", False))


@unittest.skipUnless(HAS_DEPS, "Requires torch")
class RecipePaperValuesTests(unittest.TestCase):
    def test_vitpose_b_matches_pinned_coco_config(self):
        recipe = VIPOISE_RECIPE
        self.assertEqual(recipe["optimizer"], "AdamW")
        self.assertAlmostEqual(recipe["lr"], 5e-4)
        self.assertAlmostEqual(recipe["weight_decay"], 0.1)
        self.assertEqual(list(recipe["betas"]), [0.9, 0.999])
        self.assertEqual(recipe["schedule"],
                         {"type": "vit_layer_decay", "num_layers": 12,
                          "layer_decay_rate": 0.75})
        self.assertEqual(recipe["gradient_clip_norm"], 1.0)
        schedule = recipe["lr_schedule"]
        self.assertEqual(schedule["warmup_iters"], 500)
        self.assertAlmostEqual(schedule["warmup_ratio"], 0.001)
        self.assertAlmostEqual(schedule["gamma"], 0.1)
        # step=[170, 200] of 210 epochs, rescaled to the 100-epoch budget.
        self.assertEqual(schedule["milestones_epochs"], [81, 95])

    def test_posebh_b_matches_pinned_posebh_config(self):
        recipe = POSEBH_RECIPE
        self.assertEqual(recipe["optimizer"], "AdamW")
        self.assertAlmostEqual(recipe["lr"], 1e-3)
        self.assertAlmostEqual(recipe["weight_decay"], 0.1)
        self.assertEqual(list(recipe["betas"]), [0.9, 0.999])
        self.assertEqual(recipe["schedule"],
                         {"type": "vit_layer_decay", "num_layers": 12,
                          "layer_decay_rate": 0.75})
        self.assertEqual(recipe["gradient_clip_norm"], 1.0)
        schedule = recipe["lr_schedule"]
        self.assertEqual(schedule["warmup_iters"], 500)
        self.assertAlmostEqual(schedule["warmup_ratio"], 0.001)
        self.assertEqual(schedule["milestones_epochs"], [50, 90])
        # upstream fp16=dict(loss_scale=dict(init_scale=16384.0))
        self.assertAlmostEqual(recipe["grad_scaler_init_scale"], 16384.0)

    def test_both_recipes_keep_the_shared_birdgaze_batch_and_epochs(self):
        for recipe in (VIPOISE_RECIPE, POSEBH_RECIPE):
            self.assertEqual(recipe["epochs"], 100)
            self.assertEqual(recipe["effective_batch"], 64)
            self.assertIsNone(recipe["backbone"].get("pretrained"))


if __name__ == "__main__":
    unittest.main()
