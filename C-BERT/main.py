import os

import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM

from cbert import config
from train import Trainer

import argparse
import sentencepiece as spm
class BertModel(nn.Module):
    def __init__(self, config_file='bert_config.json'):
        super(BertModel, self).__init__()
        self.config = BertConfig.from_json_file(config_file)
        self.model = BertForMaskedLM(self.config)

    # @torchsnooper.snoop()
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, labels=labels)
        return output

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    # torch.cuda.set_device(args.local_rank)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    # trainer = Trainer()
    # trainer.run()

    model = BertModel()
    # model = BertForMaskedLM(BertConfig.from_json_file('bert_config.json'))
    model.load_state_dict(torch.load('out/model_parallel_4_step_100000.ckpt'))
    encoder = model.model.base_model

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('pretraining_data/vocab.model')
    tokenizer.decode = tokenizer.Decode
    print(tokenizer.decode(['1583', 72, 70, 120, 72, 70, 6]))
    print(tokenizer.bos_id())
    print(tokenizer.eos_id())
    print(tokenizer.pad_id())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
