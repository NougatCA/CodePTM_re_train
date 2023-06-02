import os
import time
from io import StringIO
import re
import tokenize
import keyword
import logging
import json
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder, RESERVED_TOKENS

import config


def get_logger(name=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]  %(message)s')
    if name is None:
        return logging.getLogger(__name__)
    else:
        return logging.getLogger(name)


logger = get_logger()


def remove_comments_and_docstrings(source, lang):
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


def simple_split(seq: str):
    symbols = '!%^=|&*-+/\'"<>(){}[],.'
    for symbol in symbols:
        seq = seq.replace(symbol, ' ' + symbol + ' ')
    dsymbols = '=|&'
    for symbol in dsymbols:
        seq = seq.replace(symbol + '  ' + symbol, symbol + symbol)
    seq = seq.replace(';', ' ;')
    seq = re.sub(' +', ' ', seq)

    # 处理换行
    seq = seq.replace('\r\n', '\n')
    seq = seq.replace('\r', '\n')
    seq = re.sub('\n *\n', '\n', seq)
    lines = [line.strip() for line in seq.split('\n')]
    res = []
    for line in lines:
        line_tokens = []
        for token in [_ for _ in line.split(' ') if _ is not '']:
            line_tokens.extend(split_identifier(token))
        res.append(' '.join(line_tokens))
    return res


def read_case(files):
    if not isinstance(files, list):
        files = [files]
    cases = []
    error_count = 0
    success_count = 0
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line.strip())
                code_tokens = []
                for token in [_ for _ in item['code_tokens'] if _ is not '']:
                    code_tokens.extend(split_identifier(token))
                cases.append(code_tokens)
    get_logger().error(
        '{} cases are dropped because of failure when removing comments and doctrings. Get {} cases.'.format(
            error_count, success_count))
    return cases


def get_reserved_tokens():
    return RESERVED_TOKENS + ['<CLS>', '<SEP>', '<MASK>', '<UNK>']


def make_vocab(data_dir='data_collection', path='pretraining_data/vocab.txt'):
    logger = get_logger()

    # generator
    def gen(data_dir):
        for i, file in enumerate(os.listdir(data_dir)):
            logger.info('Start to reading code tokens from {}. {}/{}'.format(file, i, len(os.listdir(data_dir))))
            with open(os.path.join(data_dir, file)) as f:
                data = json.load(f)
                for case in data:
                    for token in case['code_tokens']:
                        yield token

    vocab = SubwordTextEncoder.build_from_generator(gen(data_dir), config.VOCAB_SIZE,
                                                    max_subtoken_length=config.MAX_SUBTOKEN_LENGTH,
                                                    reserved_tokens=get_reserved_tokens())
    vocab.store_to_file(path)
    logger.info('Successfully build vocab, which has been stored in {}'.format(path))
    return vocab



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
