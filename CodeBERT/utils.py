import json
import keyword
import logging
import os
import re
import time

from transformers import BertTokenizerFast

import config


def get_logger(name=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]  %(message)s')
    if name is None:
        return logging.getLogger(__name__)
    else:
        return logging.getLogger(name)


logger = get_logger()




def camel_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def is_identifier(s):
    # 内置关键字
    kw = keyword.kwlist
    # Bifs
    bifs = dir(__builtins__)
    s_list = list(s)
    # 关键字判断
    if (s in kw) | (s in bifs):
        return True
    # 数字、字母、下划线以及开头判断
    elif not s_list[0].isdigit() and sum([i.isalnum() or i == '_' for i in s_list]) == len(s_list):
        return True
    else:
        return False


def split_identifier(identifier):
    """
    Split identifier into a list of subtokens.
    Tokens except characters and digits will be eliminated.

    Args:
        identifier (str): given identifier

    Returns:
        list[str]: list of subtokens
    """
    if not is_identifier(identifier):
        return [identifier]
    words = []

    word = re.sub(r'[^a-zA-Z0-9]', ' ', identifier)
    word = re.sub(r'(\d+)', r' \1 ', word)
    split_words = word.strip().split()
    for split_word in split_words:
        camel_words = camel_split(split_word)
        for camel_word in camel_words:
            words.append(camel_word.lower())

    return words


def prepare(seq: str):
    """
    convert seq to final code seq
    :param seq: original code seq
    :return: code seq to feed into code_vocab
    """
    symbols = '!%^=|&*-+/\'"<>(){}[],.:'
    for symbol in symbols:
        seq = seq.replace(symbol, ' ' + symbol + ' ')
    dsymbols = '=|&:'
    for symbol in dsymbols:
        seq = seq.replace(symbol + '  ' + symbol, symbol + symbol)
    seq = seq.replace(';', ' ;')
    seq = re.sub(' +', ' ', seq)

    # 处理换行
    seq = seq.replace('\r\n', '\n')
    seq = seq.replace('\r', '\n')
    seq = re.sub('\n *\n', '\n', seq)
    seq = re.sub('\n', ' ', seq)
    code = ' '.join([_ for _ in seq.split(' ') if _ is not ''])
    return code

class Timer(object):
    """
    Computes elapsed time.
    """

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        self.records = [self.start]

    def record(self):
        if self.running:
            self.records.append(time.time())
            return self.records[-1] - self.records[-2]
        else:
            self.__init__()
            return 0

    def avgtime(self):
        return (self.records[-1] - self.records[0]) / (len(self.records) - 1)

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total

