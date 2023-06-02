from torch import nn
from transformers import BertConfig, BertForMaskedLM


class BertModel(nn.Module):
    def __init__(self, config_file='bert_config.json'):
        super(BertModel, self).__init__()
        self.config = BertConfig.from_json_file(config_file)
        self.model = BertForMaskedLM(self.config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, labels=labels)
        return output
