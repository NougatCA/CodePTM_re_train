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

        with open(data) as f:
            self.data = json.load(f)
        self.padding_token = self.vocab.pad_id()
        self.token_type_ids = [0 for _ in range(config.SEQUENCE_LENGTH)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s_tokens = self.data[index]
        input_ids, labels = self._random_word(s_tokens)
        assert len(input_ids) == len(labels)
        len_seq = min(len(input_ids), config.SEQUENCE_LENGTH)
        input_ids, labels = self._pad_seq(input_ids), self._pad_seq(labels)
        attention_mask = [1] * (len_seq) + [0] * (config.SEQUENCE_LENGTH - len_seq)
        return torch.from_numpy(np.array(input_ids)).long().cuda(), \
               torch.from_numpy(np.array(self.token_type_ids)).long().cuda(), \
               torch.from_numpy(np.array(attention_mask)).float().cuda(), \
               torch.from_numpy(np.array(labels)).long().cuda()

    def _random_word(self, seq):
        """mask language model, 添加15%的mask"""
        tokens = [token for token in seq]
        mask_count = 0
        input_ids = []
        labels = []
        for i, token in enumerate(tokens):
            prob = random.random()
            # 0.15的概率进行替换，包括mask, 错误，和正常
            encoded = self.vocab.EncodeAsIds(token)
            len_encoded = len(encoded)
            if prob < 0.15 and mask_count < config.MAX_MASK_NUMBER:
                mask_count += 1
                labels.extend(encoded)
                for j in range(len_encoded):
                    # 同一个token下的所有subtoken使用随机策略，如果需要用同一策略，把这行放到循环外即可
                    prob = random.random()
                    # 80% , 每一个词有80%概率（0.15*0.8）会用mask替代(sentencepiece没有mask，用bos代替)
                    if prob < 0.8:
                        input_ids.append(self.vocab.bos_id())
                    # 10%， 10%的概率随机取值
                    elif prob < 0.9:
                        input_ids.append(random.randrange(self.vocab.vocab_size()))
                    # 10%, 10%的概率正常 token
                    else:
                        input_ids.append(encoded[j])
                #
                # 正常取值
            else:
                input_ids.extend(encoded)
                labels.extend([-100 for _ in range(len_encoded)])

        return input_ids, labels

    def _pad_seq(self, seq):
        return seq[:config.SEQUENCE_LENGTH] + [self.padding_token] * (config.SEQUENCE_LENGTH - len(seq))
