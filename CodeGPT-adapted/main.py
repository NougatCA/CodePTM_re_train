import torch
from GPTCModel import BertModel
import sentencepiece as spm
from cubert import utils, config
from train import Trainer

import argparse


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
    model.load_state_dict(torch.load('out/model_parallel_2_step_220000.ckpt'))
    encoder = model.model.base_model

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('pretraining_data/vocab.model')
    print(tokenizer.Encode('Hello world'))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
