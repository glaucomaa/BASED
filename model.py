import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import tokenize


class TextDataset(Dataset):
    def __init__(self, texts, vocab, block_size=32):
        self.vocab = vocab
        self.block_size = block_size
        self.data = []
        for text in texts:
            token_ids = tokenize(text, vocab)
            for i in range(0, len(token_ids) - block_size, block_size):
                self.data.append(token_ids[i : i + block_size])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # for causal LM, input is tokens[:-1], target is tokens[1:]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# global linear and local sliding window attention.
class BasedAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, d_proj=None, window=64):
        """
        d_model: total model dimension.
        n_heads: number of attention heads.
        d_proj: projection dimension for Q/K (if None, defaults to d_model // n_heads).
        window: sliding window size for local attention.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # head dimension for value
        self.d_proj = d_proj if d_proj is not None else self.d_head
        self.window = window

        self.Wq = nn.Linear(d_model, n_heads * self.d_proj)
        self.Wk = nn.Linear(d_model, n_heads * self.d_proj)
        self.Wv = nn.Linear(d_model, n_heads * self.d_head)

    def forward(self, x):
        """
        x: [B, N, d_model]
        Returns: [B, N, d_model]
        """
        B, N, _ = x.shape
        H = self.n_heads
        device = x.device
        dtype = x.dtype

        # project and reshape for multi-head
        Q = self.Wq(x).view(B, N, H, self.d_proj).transpose(1, 2)
        K = self.Wk(x).view(B, N, H, self.d_proj).transpose(1, 2)
        V = self.Wv(x).view(B, N, H, self.d_head).transpose(1, 2)

        # we use the 2nd-order Taylor expansion:
        # phi(x) = [1, x, (x ⊗ x) / sqrt(2)]
        # dimension: d_tilde = 1 + d_proj + d_proj^2
        d_tilde = 1 + self.d_proj + self.d_proj * self.d_proj
        s = torch.zeros((B, H, d_tilde, self.d_head), device=device, dtype=dtype)
        z = torch.zeros((B, H, d_tilde), device=device, dtype=dtype)

        outputs = torch.zeros((B, H, N, self.d_head), device=device, dtype=dtype)
        sqrt_half = 1 / math.sqrt(2.0)

        # causal attention
        for i in range(N):
            q_i = Q[:, :, i, :]  # [B, H, d_proj]
            k_i = K[:, :, i, :]  # [B, H, d_proj]
            v_i = V[:, :, i, :]  # [B, H, d_head]

            # compute phi(k_i)
            one = torch.ones((B, H, 1), device=device, dtype=dtype)
            outer_kk = k_i.unsqueeze(-1) * k_i.unsqueeze(-2)  # [B, H, d_proj, d_proj]
            phi_k = torch.cat(
                [one, k_i, outer_kk.reshape(B, H, -1) * sqrt_half], dim=-1
            )  # [B, H, d_tilde]
            z = z + phi_k
            s = s + phi_k.unsqueeze(-1) * v_i.unsqueeze(-2)  # [B, H, d_tilde, d_head]

            # compute phi(q_i)
            outer_qq = q_i.unsqueeze(-1) * q_i.unsqueeze(-2)
            phi_q = torch.cat(
                [one, q_i, outer_qq.reshape(B, H, -1) * sqrt_half], dim=-1
            )  # [B, H, d_tilde]
            numerator = (phi_q.unsqueeze(-1) * s).sum(dim=-2)  # [B, H, d_head]
            denominator = (phi_q * z).sum(dim=-1, keepdim=True) + 1e-8  # [B, H, 1]
            y_linear = numerator / denominator

            # exact softmax over last window tokens
            start_idx = max(0, i - self.window + 1)
            K_win = K[:, :, start_idx : i + 1, :]  # [B, H, L, d_proj]
            V_win = V[:, :, start_idx : i + 1, :]  # [B, H, L, d_head]
            scores = (q_i.unsqueeze(-2) * K_win).sum(dim=-1)  # [B, H, L]
            weights = F.softmax(scores, dim=-1)
            y_local = (weights.unsqueeze(-1) * V_win).sum(dim=-2)  # [B, H, d_head]

            # combine outputs
            outputs[:, :, i, :] = y_linear + y_local

        # merge heads
        output = outputs.transpose(1, 2).reshape(
            B, N, H * self.d_head
        )  # [B, N, d_model]
        return output


class BasedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, window_size=8, dropout=0.1):
        super(BasedTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = BasedAttention(embed_dim, num_heads, window=window_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# model for causal language modeling
class BasedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4.0,
        block_size=32,
        window_size=8,
        dropout=0.1,
    ):
        super(BasedTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                BasedTransformerBlock(
                    embed_dim, num_heads, mlp_ratio, window_size, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        token_emb = self.token_embedding(idx)
        pos_ids = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits
