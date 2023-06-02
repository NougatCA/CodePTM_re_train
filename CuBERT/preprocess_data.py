import json
import os

import utils
from utils import logger
from fairseq.models.transformer import TransformerModel


# 把所有的数据汇集到一个文件夹下，且把code_tokens字段subword化，并把code字段删除注释
def make_data():
    languages = ['java', 'python', 'go', 'javascript', 'php', 'ruby']
    file_dirs = ['data/{}/{}/final/jsonl/train/'.format(language, language) for language in languages]
    for dir in file_dirs:
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            logger.info('collecting data from {}'.format(path))
            data = utils.read_case(path)
            with open(os.path.join('data_collection/', file), 'w') as f:
                json.dump(data, f)


def make_train_sentences(data_dir='data_collection', output_file='pretraining_data/sentence_and_next_label.json'):
    logger.info('=' * 100)
    logger.info('Start building sentence and label.')
    code_data = []
    is_next_labels = []
    for i, data_file in enumerate(os.listdir(data_dir)):
        file = os.path.join(data_dir, data_file)
        logger.info('Building sentence and label from {}, {}/{}'.format(file, i, len(os.listdir(data_dir))))
        with open(file) as f:
            data = json.load(f)
        for case in data:
            case_lines = utils.simple_split(case['code'])
            code_data.extend(case_lines)
            is_next_labels.extend([0] * (len(case_lines) - 1) + [1])
    sentence_and_next_label = {'sentence': code_data, 'next_label': is_next_labels}
    with open(output_file, 'w') as f:
        json.dump(sentence_and_next_label, f)
        logger.info('Sentence and label data are successfully built and stored in {}'.format(output_file))
    logger.info('=' * 100)


if __name__ == '__main__':
    make_data()
    # utils.make_vocab()
    make_train_sentences()
