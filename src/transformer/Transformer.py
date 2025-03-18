import pathlib
import torch
from . import Attention

class Block(torch.nn.Module):
    def __init__(self, 
                 features_n: int,
                 num_heads: int,
                 feed_forward_hidden_n: int,
                 attention_dropout: float,
                 residual_dropout: float,
                 max_sequence_length: int,
                 attention_head_type: type[torch.nn.Module] = Attention.MultiHeadAttention,
                 feed_forward_activation: torch.nn.Module = torch.nn.GELU(),
    ):
        assert features_n % num_heads == 0
        super(Block, self).__init__()
        # it will normalize features per each token
        self.norm1 = torch.nn.LayerNorm(features_n)
        self.attention = attention_head_type(
            features_n,
            features_n,
            num_heads,
            features_n // num_heads,
            attention_dropout,
            residual_dropout,
            max_sequence_length
        )
        self.norm2 = torch.nn.LayerNorm(features_n)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(features_n, feed_forward_hidden_n),
            feed_forward_activation,
            torch.nn.Linear(feed_forward_hidden_n, features_n),
        )

    def forward(self, x):
        norm_1 = self.norm1(x)
        x = x + self.attention(norm_1)
        norm_2 = self.norm2(x)
        x = x + self.feed_forward(norm_2)
        return x

class MyTransformer(torch.nn.Module):
    def __init__(self, 
                 max_sequence_length: int,
                 embeding_size: int,
                 feed_forward_hidden_n: int,
                 tokens_number: int,
                 num_heads: int,
                 residual_dropout: float,
                 attention_dropout: float,
                 embeding_dropout: float,
                 blocks_n: int,
                 attention_head_type: str = "MultiHeadAttentionFast",
    ):
        super(MyTransformer, self).__init__()
        match attention_head_type:
            case "MultiHeadAttention":
                attention_head_type_ = Attention.MultiHeadAttention
            case "MultiHeadAttentionFast":
                attention_head_type_ = Attention.MultiHeadAttentionFast
            case _:
                raise ValueError(f"Unknown attention head type: {attention_head_type}")

        self.max_sequence_length = max_sequence_length
        self.word_embeding = torch.nn.Embedding(tokens_number, embeding_size)
        self.positional_encoding = torch.nn.Embedding(max_sequence_length, embeding_size)
        self.embeding_dropout = torch.nn.Dropout(embeding_dropout)

        self.blocks = torch.nn.ModuleList([
            Block(embeding_size, num_heads, feed_forward_hidden_n, attention_dropout, residual_dropout, max_sequence_length, attention_head_type_) for _ in range(blocks_n)
        ])

        self.final_norm = torch.nn.LayerNorm(embeding_size)

    def forward(self, x):
        _, sequence_length = x.size()
        assert sequence_length <= self.positional_encoding.num_embeddings

        x = self.word_embeding(x)
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=x.device)
        positional_encoding = self.positional_encoding(positions)
        x = x + positional_encoding
        x = self.embeding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = torch.matmul(x, self.word_embeding.weight.T)
        return x
    
    def parameters_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | pathlib.Path):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str | pathlib.Path):
        self.load_state_dict(torch.load(path))
