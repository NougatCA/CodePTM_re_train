# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import logging
import argparse
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)
import multiprocessing
import time
import json
import random

from models import RetrievalModel
from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, load_and_cache_search_data
from models import get_model_size

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'vanilla': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]

    # source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    # source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # padding_length = args.block_size - len(source_ids)
    # source_ids += [tokenizer.pad_token_id] * padding_length

    source_ids = tokenizer.encode(code, max_length=args.block_size, padding='max_length', truncation=True)
    return InputFeatures(code_tokens, source_ids, js['index'], int(js['label']))


class RetrievalDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[0]

        return (torch.tensor(self.examples[i].input_ids), torch.tensor(p_example.input_ids),
                torch.tensor(n_example.input_ids), torch.tensor(label))


def evaluate(args, model, eval_data, write_to_pred=False):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_data))
    logger.info("  Num batches = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Evaluating"):
        inputs = batch[0].to(args.device)
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, vec = model(inputs, p_inputs, n_inputs, label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs = np.concatenate(vecs, 0)
    labels = np.concatenate(labels, 0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores = np.matmul(vecs, vecs.T)
    dic = {}
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        MAP.append(sum(Avep) / dic[label])

    result = {
        "eval_loss": float(perplexity),
        "eval_map": float(np.mean(MAP))
    }
    return result


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    test_dataset = RetrievalDataset(tokenizer, args, args.test_filename)

    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    vecs = []
    labels = []
    for batch in test_dataloader:
        inputs = batch[0].to(args.device)
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, vec = model(inputs, p_inputs, n_inputs, label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs = np.concatenate(vecs, 0)
    labels = np.concatenate(labels, 0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores = np.matmul(vecs, vecs.T)
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    indexs = []
    for example in test_dataset.examples:
        indexs.append(example.index)
    with open(os.path.join(args.output_dir, "predictions.jsonl"), 'w') as f:
        for index, sort_id in zip(indexs, sort_ids):
            js = {}
            js['index'] = index
            js['answers'] = []
            for idx in sort_id[:499]:
                js['answers'].append(indexs[int(idx)])
            f.write(json.dumps(js) + '\n')


eval_dataset = None


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    set_seed(args)

    args.block_size = 400

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_name_or_path == 'none':
        config = config_class.from_pretrained('roberta-base')
        model = model_class(config)
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    model = RetrievalModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    # pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.eval_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')
    loss_file = open(os.path.join(args.output_dir, 'loss.txt'), 'a+')
    time_file = open(os.path.join(args.output_dir, 'time_per_100_steps.txt'), 'a+')

    if args.do_train:
        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        # train_examples, train_data = load_and_cache_search_data(args, args.train_filename, pool, tokenizer, 'train',
        #                                                         is_sample=False)
        train_data = RetrievalDataset(tokenizer, args, args.train_filename)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_map = 0, 0
        not_map_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            last_time = time.time()
            model.train()
            for step, batch in enumerate(bar):

                inputs = batch[0].to(args.device)
                p_inputs = batch[1].to(args.device)
                n_inputs = batch[2].to(args.device)
                labels = batch[3].to(args.device)

                loss, vec = model(inputs, p_inputs, n_inputs, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                loss_file.write(f'epoch: {cur_epoch}, step {step}, global step: {global_step}, loss: {loss}\n')

                nb_tr_examples += inputs.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / nb_tr_steps, 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

                if nb_tr_steps % 100 == 0:
                    time_spend = time.time() - last_time
                    time_file.write(f'{time_spend}\n')
                    last_time = time.time()

                # if (step + 1) % save_steps == 0 and args.do_eval:

            if args.do_eval:
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                global eval_dataset
                if eval_dataset is None:
                    eval_dataset = RetrievalDataset(tokenizer, args, args.eval_filename)

                result = evaluate(args, model, eval_dataset)
                eval_map = result['eval_map']

                if args.data_num == -1:
                    tb_writer.add_scalar('dev_map', round(eval_map, 4), cur_epoch)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)

                if True or args.data_num == -1 and args.save_last_checkpoints:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_map > best_map:
                    not_map_inc_cnt = 0
                    logger.info("  Best map: %s", round(eval_map, 4))
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best map changed into %.4f\n" % (cur_epoch, round(eval_map, 4)))
                    best_map = eval_map
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-map')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.data_num == -1 or True:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_map_inc_cnt += 1
                    logger.info("map does not increase for %d epochs", not_map_inc_cnt)
                    if not_map_inc_cnt > args.patience:
                        logger.info("Early stop as map do not increase for %d times", not_map_inc_cnt)
                        fa.write("[%d] Early stop as not_map_inc_cnt=%d\n" % (cur_epoch, not_map_inc_cnt))
                        is_early_stop = True
                        break

                model.train()

            if is_early_stop:
                break

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()

        logger.info("Finish training and take %s", get_elapse_time(t0))

    loss_file.close()
    time_file.close()

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-map']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            model = model.to(args.device)

            test(args, model, tokenizer)

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
