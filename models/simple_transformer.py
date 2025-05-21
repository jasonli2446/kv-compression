import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformer(nn.Module):
    def __init__(
        self, d_model=128, nhead=4, num_layers=2, vocab_size=1000, max_seq_len=128
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, kv_cache=None):
        # input_ids: (batch, seq_len)
        positions = torch.arange(
            0, input_ids.size(1), device=input_ids.device
        ).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, d_model)
        # kv_cache is ignored in this simple version, but hook is here for extension
        x = self.transformer(x)
        x = x.transpose(0, 1)
        logits = self.lm_head(x)
        return logits, kv_cache  # Return kv_cache for compatibility
