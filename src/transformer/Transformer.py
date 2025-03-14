import pathlib
import torch
from .Attention import MultiHeadAttention

class Block(torch.nn.Module):
    def __init__(self, 
                 features_n: int,
                 num_heads: int,
                 feed_forward_hidden_n: int,
                 feed_forward_activation: torch.nn.Module = torch.nn.GELU(),
    ):
        assert features_n % num_heads == 0
        super(Block, self).__init__()
        # it will normalize features per each token
        self.norm1 = torch.nn.LayerNorm(features_n)
        self.attention = MultiHeadAttention(features_n, features_n, num_heads, features_n // num_heads)
        self.norm2 = torch.nn.LayerNorm(features_n)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(features_n, feed_forward_hidden_n),
            feed_forward_activation,
            torch.nn.Linear(feed_forward_hidden_n, features_n),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

class MyTransformer(torch.nn.Module):
    def __init__(self, 
                 max_sequence_length: int,
                 embeding_size: int,
                 feed_forward_hidden_n: int,
                 tokens_number: int,
                 num_heads: int,
                 blocks_n: int
    ):
        super(MyTransformer, self).__init__()
        self.word_embeding = torch.nn.Embedding(tokens_number, embeding_size)
        self.positional_encoding = torch.nn.Embedding(max_sequence_length, embeding_size)

        self.blocks = torch.nn.ModuleList([
            Block(embeding_size, num_heads, feed_forward_hidden_n) for _ in range(blocks_n)
        ])

        self.final_norm = torch.nn.LayerNorm(embeding_size)

    def forward(self, x):
        _, sequence_length = x.size()
        assert sequence_length <= self.positional_encoding.num_embeddings

        embeded = self.word_embeding(x)
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=x.device)
        positional_encoding = self.positional_encoding(positions)
        embeded_pos = embeded + positional_encoding

        processed = embeded_pos
        for block in self.blocks:
            processed = block(processed)

        final_norm = self.final_norm(processed)
        per_token_values = torch.matmul(final_norm, self.word_embeding.weight.T)
        apply_softmax = torch.nn.functional.softmax(per_token_values, dim=-1)
        return apply_softmax

    def save(self, path: str | pathlib.Path):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str | pathlib.Path):
        self.load_state_dict(torch.load(path))
