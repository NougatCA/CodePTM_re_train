import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class BertModel(nn.Module):
    def __init__(self, pad_token_id = 0):
        super(BertModel, self).__init__()
        self.config = GPT2Config(
            vocab_size=50000,
            n_positions=512,
            n_embd=512,
            n_layer=6,
            n_head=8,
            pad_token_id=pad_token_id
        )
        self.model = GPT2LMHeadModel(self.config)

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output
