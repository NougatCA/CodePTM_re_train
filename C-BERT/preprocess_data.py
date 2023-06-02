import json
import os

import utils
from utils import logger


# 把所有的数据汇集到一个文件夹下，且把code_tokens字段subword化，并把code字段删除注释
def make_data():
    languages = ['java', 'python', 'go', 'javascript', 'php', 'ruby']
    file_dirs = ['../data/{}/{}/final/jsonl/train/'.format(language, language) for language in languages]
    for dir in file_dirs:
        for i, file in enumerate(os.listdir(dir)):
            path = os.path.join(dir, file)
            logger.info('collecting data from {}. {}/{}'.format(path, i, len(os.listdir(dir))))
            data = utils.read_case(path)
            with open(os.path.join('data_collection/', file), 'w') as f:
                json.dump(data, f)


def make_train_sentences(data_dir='data_collection', output_file='pretraining_data/sentences.json'):
    logger.info('=' * 100)
    logger.info('Start building sentence and label.')
    code_data = []
    for i, data_file in enumerate(os.listdir(data_dir)):
        file = os.path.join(data_dir, data_file)
        logger.info('Building sentence and label from {}, {}/{}'.format(file, i, len(os.listdir(data_dir))))
        with open(file) as f:
            data = json.load(f)
        for case in data:
            code_data.append(case)
    with open(output_file, 'w') as f:
        json.dump(code_data, f)
        logger.info('Train data are successfully built and stored in {}'.format(output_file))
    logger.info('=' * 100)

def make_vocab_train_data(data_dir='data_collection', output_file='pretraining_data/vocab_train_data.txt'):
    with open(output_file, 'w') as f:
        for i, data_file in enumerate(os.listdir(data_dir)):
            file = os.path.join(data_dir, data_file)
            logger.info('Building vocab train data from {}, {}/{}'.format(file, i, len(os.listdir(data_dir))))
            with open(file) as inf:
                data = json.load(inf)
            for case in data:
                f.write(' '.join(case) + '\n')

    logger.info('Train data are successfully built and stored in {}'.format(output_file))
    logger.info('=' * 100)


if __name__ == '__main__':
    # make_data()
    # utils.make_vocab()
    make_train_sentences()
    make_vocab_train_data()
