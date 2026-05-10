import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Projection layer maps the image embeddings into LLM embedding space .
    """

    def __init__(self, vis_dim: int, llm_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(vis_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() # keeping projector in fp32 for stability, even if vision encoder is in bf16 (safer for training)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x