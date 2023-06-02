import json
import os

import numpy as np
import torch.nn
from torch import Tensor
from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaConfig

import torch.utils.data as Data
import config
from data import make_data, make_train_sentences, MLMDataset, RTDDataset
from vocab import make_vocab
from train import MLMTrainer, RTDTrainer, RobertaForPreTraining

if __name__ == '__main__':
    # make_data()
    # make_vocab()
    # make_train_sentences()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    # trainer = MLMTrainer()
    # trainer.run()
    # generator = MLMTrainer.load_model()
    # trainer = RTDTrainer(generator)
    # trainer.run()
    # data_path = os.path.join(config.PATH_TRAIN_DATA, 'MLM_data.json')
    # with open(data_path) as f:
    #     data = json.load(f)
    # code_data = data['code'][1]
    # docstring_data = data['docstring'][1]
    # print(code_data)
    # print(docstring_data)
    #
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    # model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    # model = MLMTrainer.load_model()
    # dataset = RTDDataset()
    # dataset.make_rtd_data()
    # dataset.reload_data()
    # code, attention_mask, label = dataset[0]
    # print(code)
    # inputs = tokenizer(code_data)
    # initial_input_ids = inputs['input_ids']
    # input_ids = mask(initial_input_ids)
    #
    # print('initial inputs: ', initial_input_ids)
    # print('inputs: ', input_ids)
    # # print('inputs shape: ', input_ids.shape)
    #
    # input_ids = torch.from_numpy(np.array([input_ids]))
    # outputs = model(input_ids=input_ids)
    # loss = outputs.loss
    # logits = outputs.logits
    #
    # print('outputs: ', outputs)
    # # print('output shape: ', logits.shape)
    # print(torch.argmax(logits, dim=-1)[0])
    # print(tokenizer.decode(torch.argmax(logits, dim=-1)[0]))
    # out = torch.argmax(logits, dim=-1)[0]
    # out[0] = tokenizer.cls_token_id
    # out[-1] = tokenizer.eos_token_id
    #
    # initial_input_ids = torch.from_numpy(np.array(initial_input_ids))
    # RTDLabel = 1 * (initial_input_ids != out)
    # print('initial string: ', code_data)
    # print('out string: ', tokenizer.decode(out))
    # print(initial_input_ids)
    # print(out)
    # print(RTDLabel)
    # print(initial_input_ids.shape[0])
    #
    # print(tokenizer.convert_ids_to_tokens([50118, 6755]))
    config = RobertaConfig.from_json_file('bert_config.json')
    generator = RobertaForMaskedLM(config)
    config = RobertaConfig.from_json_file('bert_config.json')
    model = RobertaForPreTraining(config, generator.base_model)
    model.load_state_dict(torch.load('rtdout/RTD_model_parallel_2_step_1000000.ckpt'))
    encoder = model.roberta

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# private static function normalizeHeaderName ( string $name ) : string { if ( 0 == \ preg_match ('/ ^ [ ^ \x00 - \x1F : ] + $ / ', $name ) ) { throw new InvalidArgumentException ( sprintf ('%s is not a valid header name. ', $name ) ) ; } return strtolower ( $name ) ; }
