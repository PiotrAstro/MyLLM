import torch 

class MultiHeadAttentionFast(torch.nn.Module):
    def __init__(self, 
                 input_features: int, 
                 output_features: int, 
                 num_heads: int, 
                 QK_features_n: int, 
                 attention_dropout: float, 
                 residual_dropout: float,
                 max_sequence_length: int,
                 ):
        assert output_features % num_heads == 0
        super(MultiHeadAttentionFast, self).__init__()
        self.QK_features_n = QK_features_n
        self.output_features_single_head = output_features // num_heads
        self.heads_number = num_heads

        self.QKV_all_heads = torch.nn.Linear(input_features, (QK_features_n * 2 + self.output_features_single_head) * num_heads)
        self.linear_projection = torch.nn.Linear(output_features, output_features)

        self.attention_dropout = torch.nn.Dropout(attention_dropout)
        self.residual_dropout = torch.nn.Dropout(residual_dropout)

        self.register_buffer("causal_mask", get_causal_mask(max_sequence_length))  # It will be automatically moved to correct device thanks to `register_buffer`

    def forward(self, x):
        sequence_length = x.size(1)
        QKV = self.QKV_all_heads(x)

        Q_length = self.QK_features_n * self.heads_number
        Q = QKV[:, :, :Q_length]
        K_length = 2 * Q_length
        K = QKV[:, :, Q_length:K_length]
        V = QKV[:, :, K_length:]

        Q = Q.reshape(-1, sequence_length, self.heads_number, self.QK_features_n).transpose(1, 2)
        K = K.reshape(-1, sequence_length, self.heads_number, self.QK_features_n).transpose(1, 2)
        V = V.reshape(-1, sequence_length, self.heads_number, self.output_features_single_head).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1))
        attention *= (1.0 / self.QK_features_n ** 0.5)
        attention += self.causal_mask[:sequence_length, :sequence_length]  # type: ignore
        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        final = torch.matmul(attention, V)
        final = final.transpose(1, 2)
        final = final.reshape(-1, sequence_length, self.output_features_single_head * self.heads_number)

        final_projected = self.linear_projection(final)
        final_projected = self.residual_dropout(final_projected)
        return final_projected



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_features: int, 
                 output_features: int, 
                 num_heads: int, 
                 QK_features_n: int, 
                 attention_dropout: float, 
                 residual_dropout: float,
                 max_sequence_length: int,
                 ):
        assert output_features % num_heads == 0
        super(MultiHeadAttention, self).__init__()
        self.attention_heads = torch.nn.ModuleList([
            AttentionHead(input_features, output_features // num_heads, QK_features_n, attention_dropout, max_sequence_length) for _ in range(num_heads)
        ])
        self.linear_projection = torch.nn.Linear(output_features, output_features)
        self.residual_dropout = torch.nn.Dropout(residual_dropout)

    def forward(self, x):
        final = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        final_projected = self.linear_projection(final)
        final_projected = self.residual_dropout(final_projected)
        return final_projected


def get_causal_mask(size: int):
    mask = torch.ones(size, size)
    mask = torch.tril(mask)
    mask = mask.masked_fill(mask == 0, float('-inf'))
    return mask

class AttentionHead(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, QK_features_n: int, attention_dropout: float, max_sequence_length: int):
        super(AttentionHead, self).__init__()
        self.Q_dense = torch.nn.Linear(input_features, QK_features_n)
        self.K_dense = torch.nn.Linear(input_features, QK_features_n)
        self.V_dense = torch.nn.Linear(input_features, output_features)
        self.attention_dropout = torch.nn.Dropout(attention_dropout)
        self.register_buffer("causal_mask", get_causal_mask(max_sequence_length))

    def forward(self, x):
        sequence_length = x.size(1)

        Q = self.Q_dense(x)
        K = self.K_dense(x)
        V = self.V_dense(x)

        attention = torch.matmul(Q, K.transpose(1, 2))
        attention = attention / (Q.size(-1) ** 0.5)  # scaling

        attention += self.causal_mask[:sequence_length, :sequence_length]  # type: ignore
        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        out = torch.matmul(attention, V)
        return out
