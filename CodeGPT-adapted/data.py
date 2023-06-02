import json
import random

import numpy as np
from torch.utils.data import Dataset
import utils
import config
import torch

import sentencepiece as spm

class CodeDataset(Dataset):

    def __init__(self, data=config.DATA_PATH, vocab=config.VOCAB_PATH):

        sp = spm.SentencePieceProcessor()
        sp.Load(vocab)
        self.vocab = sp
        utils.logger.info('vocab size: {}'.format(self.vocab.vocab_size()))

        with open(data) as f:
            self.data = json.load(f)
        self.padding_token = self.vocab.pad_id()
        self.token_type_ids = [0 for _ in range(config.SEQUENCE_LENGTH)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.vocab.EncodeAsIds(' '.join(self.data[index]))
        labels = input_ids
        len_seq = min(len(input_ids), config.SEQUENCE_LENGTH)
        input_ids, labels = self._pad_seq(input_ids), self._pad_seq(labels)
        attention_mask = [1] * (len_seq) + [0] * (config.SEQUENCE_LENGTH - len_seq)
        return torch.from_numpy(np.array(input_ids)).long().cuda(), \
               torch.from_numpy(np.array(attention_mask)).float().cuda(), \
               torch.from_numpy(np.array(labels)).long().cuda()

    def _pad_seq(self, seq):
        return seq[:config.SEQUENCE_LENGTH] + [self.padding_token] * (config.SEQUENCE_LENGTH - len(seq))
