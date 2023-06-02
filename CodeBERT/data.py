import json
import os
import random
import re
import tokenize
import time
from io import StringIO

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig

import config
import utils
from utils import logger
from torch.utils.data import Dataset


def _remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def _read_case(files):
    if not isinstance(files, list):
        files = [files]
    cases = []
    error_count = 0
    success_count = 0
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                try:
                    cases.append({
                        'code': _remove_comments_and_docstrings(item['code'], item['language']),
                        'code_tokens': [_ for _ in item['code_tokens'] if _ is not ''],
                        'docstring': item['docstring'],
                        'docstring_tokens': item['docstring_tokens']}
                    )
                    success_count += 1
                except:
                    error_count += 1
    logger.info(
        '{} cases are dropped because of failure when removing comments and doctrings. Get {} cases.'.format(
            error_count, success_count))
    return cases


# 收集数据{'code','code_tokens','docstring','docstring_tokens'}，保存到 config.PATH_DATA_COLLECTION 目录下
def make_data():
    languages = ['java', 'python', 'go', 'javascript', 'php', 'ruby']
    file_dirs = ['{}/{}/{}/final/jsonl/train/'.format(config.PATH_DATA_SOURCE, language, language) for language in
                 languages]
    for dir in file_dirs:
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            logger.info('collecting data from {}'.format(path))
            data = _read_case(path)
            with open(os.path.join('data_collection/', file), 'w') as f:
                json.dump(data, f)


def make_train_sentences(data_dir=config.PATH_DATA_COLLECTION, output_dir=config.PATH_TRAIN_DATA):
    logger.info('=' * 100)
    logger.info('Start building data for MLM.')
    code_data = []
    docstring_data = []
    for i, data_file in enumerate(os.listdir(data_dir)):
        file = os.path.join(data_dir, data_file)
        logger.info('Building MLM data from {}, {}/{}'.format(file, i, len(os.listdir(data_dir))))
        with open(file) as f:
            data = json.load(f)
        for case in data:
            if 'docstring' in dict(case).keys() and case['docstring'].strip() is not '':
                code = utils.prepare(case['code'])
                code_data.append(code)
                docstring = utils.prepare(case['docstring'])
                docstring_data.append(docstring)
    MLM_data = {'code': code_data, 'docstring': docstring_data}
    MLM_path = os.path.join(output_dir, 'MLM_data.json')
    with open(MLM_path, 'w') as f:
        json.dump(MLM_data, f)
        logger.info('MLM data are successfully built and stored in {}'.format(MLM_path))

    logger.info('Start building data for RTD.')
    code_data = []
    for i, data_file in enumerate(os.listdir(data_dir)):
        file = os.path.join(data_dir, data_file)
        logger.info('Building RTD data from {}, {}/{}'.format(file, i, len(os.listdir(data_dir))))
        with open(file) as f:
            data = json.load(f)
        for case in data:
            code = utils.prepare(case['code'])
            code_data.append(code)
    RTD_data = {'code': code_data}
    RTD_path = os.path.join(output_dir, 'RTD_data.json')
    with open(RTD_path, 'w') as f:
        json.dump(RTD_data, f)
        logger.info('RTD data are successfully built and stored in {}'.format(RTD_path))

    logger.info('=' * 100)


class MLMDataset(Dataset):
    def __init__(self,
                 data_path=os.path.join(config.PATH_TRAIN_DATA, 'MLM_data.json'),
                 vocab_path=config.PATH_VOCAB
                 ):

        with open(data_path) as f:
            data = json.load(f)
        self.code_data = data['code']
        self.docstring_data = data['docstring']
        assert len(self.code_data) == len(self.docstring_data)
        self.vocab = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        logger.info('vocab size: {}'.format(self.vocab.vocab_size))
        self.padding_token_id = self.vocab.pad_token_id
        self.CLS_token_id = self.vocab.cls_token_id
        self.SEP_token_id = self.vocab.sep_token_id
        self.EOS_token_id = self.vocab.eos_token_id
        self.MSK_token_id = self.vocab.mask_token_id
        self.token_type_ids = [0 for _ in range(config.LEN_DOCSTRING_SEQ + 2)] \
                              + [1 for _ in range(config.LEN_CODE_SEQ + 1)]

    def __len__(self):
        return len(self.code_data)

    def __getitem__(self, index):
        code_, docstring_ = self._get_sentence(index)
        code = self.vocab(code_, add_special_tokens=False, max_length=1024, truncation=True)['input_ids']
        docstring = self.vocab(docstring_, add_special_tokens=False, max_length=1024, truncation=True)['input_ids']
        len_code = min(len(code), config.LEN_CODE_SEQ)
        len_docstring = min(len(docstring), config.LEN_DOCSTRING_SEQ)
        code, code_label = self._random_word(code)
        docstring, docstring_label = self._random_word(docstring)
        code = self._pad_seq(code, config.LEN_CODE_SEQ)
        code_label = self._pad_seq(code_label, config.LEN_CODE_SEQ)
        docstring = self._pad_seq(docstring, config.LEN_DOCSTRING_SEQ)
        docstring_label = self._pad_seq(docstring_label, config.LEN_DOCSTRING_SEQ)

        input_ids = [self.CLS_token_id] + docstring + [self.SEP_token_id] + code + [self.EOS_token_id]
        labels = [self.padding_token_id] + docstring_label + [self.padding_token_id] + code_label + [
            self.padding_token_id]
        attention_mask = [1] * (1 + len_docstring) + [0] * (config.LEN_DOCSTRING_SEQ - len_docstring) + [1] * (
                1 + len_code) + [0] * (config.LEN_CODE_SEQ - len_code) + [1]
        assert len(input_ids) == 512 and len(labels) == 512 and len(attention_mask) == 512 and len(
            self.token_type_ids) == 512

        return torch.from_numpy(np.array(input_ids)).long().cuda(), \
               torch.from_numpy(np.array(self.token_type_ids)).long().cuda(), \
               torch.from_numpy(np.array(attention_mask)).float().cuda(), \
               torch.from_numpy(np.array(labels)).long().cuda()

    def _get_sentence(self, index):
        return self.code_data[index], self.docstring_data[index]

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
                    tokens[i] = self.MSK_token_id
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

    def _pad_seq(self, seq, seq_len):
        return seq[:seq_len] + [self.padding_token_id] * (seq_len - len(seq))


class RTDDataset(Dataset):
    def __init__(self,
                 data_path=os.path.join(config.PATH_TRAIN_DATA, 'RTD_data.json'),
                 MLMmodel_path=os.path.join(config.MODEL_SAVE_PATH, config.PATH_SAVE_MODEL_MLMModel),
                 mode='train'
                 ):
        assert mode in ['train', 'eval', 'test', 'preprocess']

        if mode is 'preprocess':
            with open(data_path) as f:
                data = json.load(f)
            self.code_data = data['code']

            model = RobertaForMaskedLM(RobertaConfig.from_json_file(config.PATH_MODEL_CONFIG))
            model.load_state_dict(torch.load(MLMmodel_path))
            self.generator = model.cuda()

        self.vocab = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        # self.generator: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base")
        logger.info('vocab size: {}'.format(self.vocab.vocab_size))
        self.padding_token_id = self.vocab.pad_token_id
        self.CLS_token_id = self.vocab.cls_token_id
        self.SEP_token_id = self.vocab.sep_token_id
        self.EOS_token_id = self.vocab.eos_token_id
        self.MSK_token_id = self.vocab.mask_token_id
        self.token_type_ids = [1 for _ in range(config.LEN_CODE_SEQ + 2)]
        self.reload_data()

    def __getitem__(self, index):
        code, label = self.data[index]
        len_code = code.shape[0]
        len_label = code.shape[0]
        assert len_label == len_code
        attention_mask = [1] * len_code + [0] * (config.LEN_CODE_SEQ - len_code)
        code = torch.cat([code, torch.from_numpy(np.array([self.padding_token_id] * (config.LEN_CODE_SEQ - len_code)))],
                         0)
        label = torch.cat([label, torch.from_numpy(np.array([0] * (config.LEN_CODE_SEQ - len_code)))], 0)
        return code.long().cuda(), torch.from_numpy(np.array(attention_mask)).float().cuda(), label.long().cuda()

    def __len__(self):
        if self.data:
            return len(self.data)
        if self.code_data:
            return len(self.code_data)
        return 0

    def make_rtd_data(self):
        rtd_data = []
        t = time.time()
        for i, code in enumerate(self.code_data):
            initial_code = self.vocab(code, max_length=1024, truncation=True)['input_ids'][:config.LEN_CODE_SEQ]

            code, code_label = self._random_word(initial_code)

            input_ids = torch.from_numpy(np.array([code])).cuda()
            code_generated: torch.Tensor = self.generator(input_ids=input_ids).logits
            out = torch.argmax(code_generated, dim=-1)[0].cpu()
            out[0] = self.vocab.cls_token_id
            out[-1] = self.vocab.eos_token_id
            initial_code = torch.from_numpy(np.array(initial_code))
            RTDLabel = 1 * (initial_code != out)
            rtd_data.append([out, RTDLabel])
            if i % 1000 == 0:
                logger.info('finished {}/{}, time: {:.2f}'.format(i, len(self.code_data), time.time() - t))

        torch.save(rtd_data, config.PATH_TRAIN_DATA_RTD)

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
                    tokens[i] = self.MSK_token_id
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

    def _pad_seq(self, seq, seq_len):
        return seq[:seq_len] + [self.padding_token_id] * (seq_len - len(seq))

    def reload_data(self, data_path=config.PATH_TRAIN_DATA_RTD):
        if os.path.exists(data_path):
            self.data = torch.load(data_path)
        else:
            logger.error('rtd data not found: {}'.format(data_path))
            logger.info('Invoke make_rtd_data() first.')
