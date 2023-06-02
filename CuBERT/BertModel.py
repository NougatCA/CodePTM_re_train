from transformers import BertConfig, BertForPreTraining
import torch.nn as nn


class BertModel(nn.Module):
    def __init__(self, config_file='bert_config.json'):
        super(BertModel, self).__init__()
        self.config = BertConfig.from_json_file(config_file)
        self.model = BertForPreTraining(self.config)

    # @torchsnooper.snoop()
    def forward(self, input_ids, token_type_ids, attention_mask, labels, next_sentence_label):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            labels=labels, next_sentence_label=next_sentence_label)
        return output
