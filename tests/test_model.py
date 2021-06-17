import pytest
import torch

from src.models.model_conv32 import ConvVAE as ConvVAE32
from src.models.model_conv224 import ConvVAE as ConvVAE224


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("torch.randn((1, 1, 33, 33))", torch.Size((1, 2, 33, 33))),
        ("torch.randn((2, 1, 33, 33))", torch.Size((2, 2, 33, 33))),
        ("torch.randn((1, 1, 10, 10))", torch.Size((1, 2, 33, 33))),
        ("torch.randn((1, 1, 128, 128))", torch.Size((1, 2, 33, 33))),
        ("torch.randn((1, 1, 128, 32))", torch.Size((1, 2, 33, 33))),
    ],
)
def test_conv_32_dims(test_input, expected):
    model = ConvVAE32()
    recon, _, _ = model(eval(test_input))
    assert recon.shape == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("torch.randn((1, 1, 224, 224))", torch.Size((1, 2, 224, 224))),
        ("torch.randn((2, 1, 224, 224))", torch.Size((2, 2, 224, 224))),
        ("torch.randn((1, 1, 512, 512))", torch.Size((1, 2, 224, 224))),
        ("torch.randn((1, 1, 32, 32))", torch.Size((1, 2, 224, 224))),
        ("torch.randn((1, 1, 128, 32))", torch.Size((1, 2, 224, 224))),
    ],
)
def test_conv_224_dims(test_input, expected):
    model = ConvVAE224()
    recon, _, _ = model(eval(test_input))
    assert recon.shape == expected
