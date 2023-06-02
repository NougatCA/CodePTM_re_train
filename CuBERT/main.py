import torch
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from torch import nn
from transformers import BertConfig, BertForPreTraining

from cubert import utils, config
from train import Trainer

import argparse

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

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)
    # trainer = Trainer()
    # trainer.run()
    # vocab = utils.load_vocab('vocab.txt')
    # print(vocab.encode_without_tokenizing(config.PAD_TOKEN))
    model = BertModel()
    model.load_state_dict(torch.load('out/model_parallel_2_step_110000.ckpt'))
    encoder = model.model.base_model

    tokenizer = SubwordTextEncoder('vocab.txt')
    print(tokenizer.encode(''))
    print(tokenizer.encode_without_tokenizing('<pad>'))
    print(tokenizer.encode_without_tokenizing('<CLS>'))
    print(tokenizer.encode_without_tokenizing('<SEP>'))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
