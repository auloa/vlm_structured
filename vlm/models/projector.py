import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Projection layer maps the image embeddings into LLM embedding space .
    """

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return self.net(x.float()).to(orig_dtype)