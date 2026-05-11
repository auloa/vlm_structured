import torch
import torch.nn as nn


class Projector(nn.Module):
    """Maps image embeddings into the LLM embedding space."""

    def __init__(self, vis_dim: int, llm_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(vis_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class AttentionPool(nn.Module):
    def __init__(self, dim, num_queries=64, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                     # x: (B, N_visual, dim)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out + q)