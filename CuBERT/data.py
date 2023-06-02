import json
import random

import numpy as np
from torch.utils.data import Dataset
import utils
import config
import torch


class CodeDataset(Dataset):

    def __init__(self, data='pretraining_data/sentence_and_next_label.json', vocab='vocab.txt', mode='train'):
        assert mode in ['train', 'valid', 'test']
        self.vocab = utils.load_vocab(vocab)
        utils.logger.info('vocab size: {}'.format(self.vocab.vocab_size))
        with open(data) as f:
            self.data = json.load(f)
        self.code_data = self.data['sentence']
        self.is_next_labels = self.data['next_label']
        assert len(self.code_data) == len(self.is_next_labels)
        self.padding_token = self.vocab.encode_without_tokenizing(config.PAD_TOKEN)
        self.CLS_token = self.vocab.encode_without_tokenizing(config.CLS_TOKEN)
        self.SEP_token = self.vocab.encode_without_tokenizing(config.SEP_TOKEN)
        self.token_type_ids = [0 for _ in range(config.SEQUENCE_LENGTH + 2)] + [1 for _ in
                                                                                range(config.SEQUENCE_LENGTH)]

    def __len__(self):
        return len(self.code_data)

    def __getitem__(self, index):
        s1, s2, is_next_label = self._get_sentences(index)
        s1 = self.vocab.encode(s1)
        s2 = self.vocab.encode(s2)
        len_s1, len_s2 = min(len(s1), config.SEQUENCE_LENGTH), min(len(s2), config.SEQUENCE_LENGTH)
        s1, label1 = self._random_word(s1)
        s1, label1 = self._pad_seq(s1), self._pad_seq(label1)
        s2, label2 = self._random_word(s2)
        s2, label2 = self._pad_seq(s2), self._pad_seq(label2)

        input_ids = self.CLS_token + s1 + self.SEP_token + s2
        labels = self.padding_token + label1 + self.padding_token + label2
        attention_mask = [1] * (1 + len_s1) + [0] * (config.SEQUENCE_LENGTH - len_s1) + [1] * (1 + len_s2) + [0] * (
                config.SEQUENCE_LENGTH - len_s2)
        if (len(input_ids) != 512) or (len(attention_mask) != 512) or (len(labels) != 512):
            print('stop here.')
        return torch.from_numpy(np.array(input_ids)).long().cuda(), \
               torch.from_numpy(np.array(self.token_type_ids)).long().cuda(), \
               torch.from_numpy(np.array(attention_mask)).float().cuda(), \
               torch.from_numpy(np.array(labels)).long().cuda(), \
               torch.from_numpy(np.array(is_next_label)).long().cuda()

    # return sentences string
    def _get_sentences(self, index):
        prob = random.random()
        if prob < 0.5:
            return self.code_data[index], self.code_data[random.randrange(len(self.code_data))], 1
        return self.code_data[index], self.code_data[(index + 1) % len(self.code_data)], self.is_next_labels[index]

    def _random_word(self, seq):
        """mask language model, 添加15%的mask"""
        tokens = [token for token in seq]
        mask_count = 0
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            # 0.15的概率进行替换，包括mask, 错误，和正常
            if prob < 0.15 and mask_count < config.MAX_MASK_NUMBER:
                mask_count += 1
                prob /= 0.15
                # 80% , 每一个词有80%概率（0.15*0.8）会用mask替代
                if prob < 0.8:
                    tokens[i] = self.vocab.encode_without_tokenizing(config.MASK_TOKEN)[0]
                # 10%， 10%的概率随机取值
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab.vocab_size)
                # 10%, 10%的概率正常token
                else:
                    tokens[i] = token
                #
                output_label.append(token)
            # 正常取值
            else:
                tokens[i] = token
                output_label.append(-100)
        return tokens, output_label

    def _pad_seq(self, seq):
        return seq[:config.SEQUENCE_LENGTH] + self.padding_token * (config.SEQUENCE_LENGTH - len(seq))
