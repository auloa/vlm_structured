import pytest
import torch
from vlm.models.projector import Projector


@pytest.fixture
def projector():
    return Projector(vis_dim=768, llm_dim=2048)


def test_output_shape(projector):
    dummy = torch.randn(1, 197, 768)
    output = projector(dummy)
    assert output.shape == torch.Size([1, 197, 2048])


def test_batch_dimension(projector):
    dummy = torch.randn(4, 197, 768)
    output = projector(dummy)
    assert output.shape == torch.Size([4, 197, 2048])


def test_trainable_params(projector):
    trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    assert trainable == 6_299_648


def test_output_dtype_matches_input(projector):
    dummy = torch.randn(1, 197, 768, dtype=torch.bfloat16)
    projector = Projector(vis_dim=768, llm_dim=2048)
    output = projector(dummy.float())
    assert output.dtype == torch.float32