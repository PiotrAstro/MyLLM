# impor cache from functools
from functools import lru_cache
import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, num_heads: int, QK_features_n: int):
        assert output_features % num_heads == 0
        super(MultiHeadAttention, self).__init__()
        self.attention_heads = torch.nn.ModuleList([
            AttentionHead(input_features, output_features // num_heads, QK_features_n) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.attention_heads], dim=-1)

@lru_cache(maxsize=None)
def get_causal_mask(size: int, device):
    mask = torch.ones(size, size, device=device)
    mask = torch.tril(mask)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    return mask

class AttentionHead(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, QK_features_n: int):
        super(AttentionHead, self).__init__()
        self.Q_dense = torch.nn.Linear(input_features, QK_features_n)
        self.K_dense = torch.nn.Linear(input_features, QK_features_n)
        self.V_dense = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        Q = self.Q_dense(x)
        K = self.K_dense(x)
        V = self.V_dense(x)

        attention = torch.matmul(Q, K.transpose(1, 2))
        attention = attention / (Q.size(-1) ** 0.5)  # scaling
        attention = torch.nn.functional.softmax(attention, dim=-1)
        causal_mask = get_causal_mask(x.size(1), x.device)
        attention = attention + causal_mask 
        out = torch.matmul(attention, V)
        return out
