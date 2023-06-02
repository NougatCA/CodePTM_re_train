import json
import os
from tokenizers.models import WordLevel, WordPiece
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

import config
from utils import logger


def make_vocab():
    """
    :param keyword: ['docstring_tokens', 'code_tokens']
    """

    start_vocab = [config.TOKEN_PAD, config.TOKEN_CLS, config.TOKEN_SEP, config.TOKEN_MSK, config.TOKEN_UNK, config.TOKEN_EOS, config.TOKEN_BOS]

    # generator
    def gen():
        for i, file in enumerate(os.listdir(config.PATH_DATA_COLLECTION)):
            logger.info('Start to reading docstring tokens from {}. {}/{}'.format(file, i, len(os.listdir(config.PATH_DATA_COLLECTION))))
            with open(os.path.join(config.PATH_DATA_COLLECTION, file)) as f:
                data = json.load(f)
                for case in data:
                    items = case['code_tokens']
                    items.extend(case['docstring_tokens'])
                    for token in items:
                        yield token

    logger.info('Start to build vocab.')
    tokenizer = Tokenizer(WordPiece(unk_token=config.TOKEN_UNK))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(special_tokens=start_vocab, vocab_size=config.VOCAB_SIZE)
    tokenizer.train_from_iterator(gen(), trainer=trainer)
    tokenizer.save(config.PATH_VOCAB)
    logger.info('Vocab is successfully built and stored in {}.'.format(config.PATH_VOCAB))


