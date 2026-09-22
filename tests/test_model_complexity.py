import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] / 'Bird_HKE'
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from lib.utilities.model_complexity import profile_model_complexity
    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    'PyTorch is not installed in this environment',
)
class ModelComplexityTests(unittest.TestCase):

    def test_grouped_convolution_does_not_divide_channels_twice(self):
        model = nn.Conv2d(4, 4, kernel_size=3, groups=4, bias=False)
        result = profile_model_complexity(model, torch.zeros(1, 4, 5, 5))
        self.assertEqual(result.macs_by_operation['convolution'], 324)

    def test_linear_counts_every_batch_and_token_vector(self):
        model = nn.Linear(4, 3, bias=False)
        result = profile_model_complexity(model, torch.zeros(2, 5, 4))
        self.assertEqual(result.macs_by_operation['linear_projection'], 120)

    def test_transposed_convolution_uses_input_scatter_work(self):
        model = nn.ConvTranspose2d(
            4, 6, kernel_size=2, stride=2, groups=2, bias=False
        )
        result = profile_model_complexity(model, torch.zeros(1, 4, 3, 3))
        expected = 1 * 4 * 3 * 3 * (6 // 2) * 2 * 2
        self.assertEqual(
            result.macs_by_operation['transposed_convolution'], expected
        )

    def test_attention_adds_qk_and_av_products(self):
        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.qkv = nn.Linear(4, 12, bias=False)
                self.proj = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                batch, tokens, channels = x.shape
                qkv = self.qkv(x).reshape(
                    batch, tokens, 3, self.num_heads, channels // self.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, key, value = qkv.unbind(0)
                attention = torch.softmax(q @ key.transpose(-2, -1), dim=-1)
                output = attention @ value
                output = output.transpose(1, 2).reshape(batch, tokens, channels)
                return self.proj(output)

        result = profile_model_complexity(
            TinyAttention(), torch.zeros(1, 3, 4)
        )
        self.assertEqual(result.macs_by_operation['linear_projection'], 192)
        self.assertEqual(result.macs_by_operation['attention_qk_av'], 72)
        self.assertEqual(result.total_macs, 264)

    def test_mamba_vision_mixer_is_counted_atomically(self):
        class MambaVisionMixer(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Linear(4, 4, bias=False)
                self.x_proj = nn.Linear(2, 7, bias=False)
                self.dt_proj = nn.Linear(1, 2, bias=False)
                self.out_proj = nn.Linear(4, 4, bias=False)
                self.conv1d_x = nn.Conv1d(2, 2, 3, groups=2, bias=False)
                self.conv1d_z = nn.Conv1d(2, 2, 3, groups=2, bias=False)
                self.A_log = nn.Parameter(torch.zeros(2, 3))
                self.D = nn.Parameter(torch.ones(2))

            def forward(self, x):
                return x

        result = profile_model_complexity(
            MambaVisionMixer(), torch.zeros(1, 5, 4)
        )
        self.assertEqual(result.macs_by_operation['mamba_projection'], 240)
        self.assertEqual(
            result.macs_by_operation['mamba_causal_convolution'], 60
        )
        self.assertEqual(result.macs_by_operation['mamba_selective_scan'], 280)
        self.assertEqual(result.total_macs, 580)

    def test_parameter_and_flop_conventions_are_explicit(self):
        model = nn.Linear(4, 3)
        model.bias.requires_grad = False
        result = profile_model_complexity(model, torch.zeros(1, 4))
        self.assertEqual(result.total_parameters, 15)
        self.assertEqual(result.trainable_parameters, 12)
        self.assertEqual(result.non_trainable_parameters, 3)
        self.assertEqual(result.gflops, 2.0 * result.gmacs)
        payload = result.to_dict()
        self.assertIn('counting_convention', payload)
        self.assertEqual(payload['params_M'], 12 / 1e6)


if __name__ == '__main__':
    unittest.main()
