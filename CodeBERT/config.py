# 经过整理的数据位置，用于构建词表
PATH_DATA_COLLECTION = 'data_collection'
# 原始数据位置
PATH_DATA_SOURCE = '../data'
# 词表存储目录
PATH_VOCAB = 'vocab/vocab.json'
# 训练数据存储目录
PATH_TRAIN_DATA = 'train_data'
# rtd训练数据存储目录
PATH_TRAIN_DATA_RTD = 'train_data/rtd.pth'
# 模型保存目录
PATH_SAVE_MODEL = 'model/'
# 生成器模型名
PATH_SAVE_MODEL_MLMModel = 'MLMmodel_parallel_2_step_30000.ckpt'
# 模型配置文件
PATH_MODEL_CONFIG = 'bert_config.json'

VOCAB_SIZE = 50000

MAX_MASK_NUMBER = 20

TOKEN_PAD='[PAD]'
TOKEN_CLS='[CLS]'
TOKEN_SEP='[SEP]'
TOKEN_MSK='[MASK]'
TOKEN_UNK='[UNK]'
TOKEN_EOS='[EOS]'
TOKEN_BOS='[BOS]'

LEN_CODE_SEQ = 383
LEN_DOCSTRING_SEQ = 126

LR = 5e-5
BATCH_SIZE = 16
TARGET_BATCH_SIZE = 256
NUM_WORKERS = 2
WARMUP_STEPS = 2000
TOTAL_STEPS = 1000000
MODEL_SAVE_PATH = 'model/'
SAVE_STEP = 10000
TIME_SAVE_STEP = 1000