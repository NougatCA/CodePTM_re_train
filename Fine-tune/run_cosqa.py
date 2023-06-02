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
import json
import math
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)
import multiprocessing
import time
from accelerate import Accelerator
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from models import CosQAModel
from configs import add_args, set_seed, set_dist
from utils import get_filenames, get_elapse_time, acc_and_f1
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

    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids  # code tokenized idxs
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids  # nl tokenized idxs
        self.label = label
        self.idx = idx


class InputFeaturesTrip(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """

    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTrip, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids


def convert_examples_to_features(js, tokenizer, args):
    label = js['label'] if js.get('label', None) else 0

    code = js['code']
    if args.code_type == 'code_tokens':
        code = js['code_tokens']
    # if args.model_type != "gpt2":
    #     code_tokens = tokenizer.tokenize(code)[:args.max_seq_length - 2]
    #     code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    #     code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    #     padding_length = args.max_seq_length - len(code_ids)
    #     code_ids += [tokenizer.pad_token_id] * padding_length
    # else:
    code_tokens = code.split()
    code_ids = tokenizer.encode(" ".join(code_tokens), padding='max_length', max_length=args.max_seq_length, truncation=True)

    nl = js['doc']
    # if args.model_type != "gpt2":
    #     nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length - 2]
    #     nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    #     nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    #     padding_length = args.max_seq_length - len(nl_ids)
    #     nl_ids += [tokenizer.pad_token_id] * padding_length
    # else:
    nl_tokens = nl.split()
    nl_ids = tokenizer.encode(" ".join(nl_tokens), padding='max_length', max_length=args.max_seq_length, truncation=True)

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        # file: json fiel, each dict contains keys: idx, query, doc, code (or 'function_tokens' in a list of string), docstring_tokens (list of strings)
        self.examples = []
        data = []
        with open(file_path, 'r') as f:
            data = json.load(f)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        # if training set, print first three exampls
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].code_ids), \
               torch.tensor(self.examples[i].nl_ids), \
               torch.tensor(self.examples[i].label)


eval_dataset = None


def evaluate(args, model, tokenizer, accelerator=None):
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        eval_dataset = TextDataset(tokenizer, args, args.dev_filename)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0,
                                 pin_memory=True)

    if accelerator is not None:
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_predictions = []
    all_labels = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, predictions = model(code_inputs, nl_inputs, labels)
            # lm_loss,code_vec,nl_vec = model(code_inputs,nl_inputs)
            eval_loss += lm_loss.mean().item()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        nb_eval_steps += 1
    all_predictions = torch.cat(all_predictions, 0).squeeze().numpy()
    all_labels = torch.cat(all_labels, 0).squeeze().numpy()
    eval_loss = torch.tensor(eval_loss / nb_eval_steps)

    results = acc_and_f1(all_predictions, all_labels)
    results.update({"eval_loss": float(eval_loss)})
    return results


def test(args, model, tokenizer, accelerator=None):
    test_dataset = TextDataset(tokenizer, args, args.test_filename)
    eval_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    answers = {}
    with open(args.answer_filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            answers[line.split('\t')[0]] = int(line.split('\t')[1])

    if accelerator is not None:
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    all_predictions = []
    all_labels = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, predictions = model(code_inputs, nl_inputs, labels)
            eval_loss += lm_loss.mean().item()
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

        nb_eval_steps += 1
    all_predictions = torch.cat(all_predictions, 0).squeeze().numpy()
    # all_labels = torch.cat(all_labels, 0).squeeze().numpy()
    # eval_loss = torch.tensor(eval_loss / nb_eval_steps)
    # results = acc_and_f1(all_predictions, all_labels)
    # results.update({"eval_loss": float(eval_loss)})

    # for key in results.keys():
    #     logger.info("  Final test %s = %s", key, str(results[key]))
    logger.info("  " + "*" * 20)
    predictions = {}
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(test_dataset.examples, all_predictions.tolist()):
            f.write(example.idx + '\t' + str(int(pred)) + '\n')
            predictions[example.idx] = int(pred)

    scores = calculate_scores(answers, predictions)
    for key, value in scores.items():
        logger.info("  Final test %s = %s", key, str(value))

    return scores


def calculate_scores(answers, predictions):
    y_trues, y_preds = [], []
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
        y_trues.append(answers[key])
        y_preds.append(predictions[key])
    scores = {}
    scores['precision'] = precision_score(y_trues, y_preds)
    scores['recall'] = recall_score(y_trues, y_preds)
    scores['f1'] = f1_score(y_trues, y_preds)
    scores['acc'] = accuracy_score(y_trues, y_preds)
    return scores


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(mixed_precision="fp16")
    logger.info(accelerator.state)

    set_dist(args, accelerator=accelerator)
    set_seed(args)

    args.num_train_epochs = 3
    args.learning_rate = 5e-6
    args.warmup_steps = 500
    args.max_seq_length = 200
    args.code_type = "code"

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_name_or_path == 'none':
        config = config_class.from_pretrained('roberta-base')
        config.num_labels = 2
        model = model_class(config)
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        config.num_labels = 2
        model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    model = CosQAModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(args.device)

    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    args.answer_filename = f"{args.data_dir}/{args.task}/answers.txt"

    pool = multiprocessing.Pool(cpu_cont)

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

        train_data = TextDataset(tokenizer, args, args.train_filename)

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

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        model.zero_grad()

        best_results = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "acc_and_f1": 0.0}
        global_step, best_acc = 0, 0
        not_acc_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            # for cur_epoch in range(3):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            last_time = time.time()
            model.train()
            for step, batch in enumerate(bar):
                code_inputs = batch[0].to(args.device)
                nl_inputs = batch[1].to(args.device)
                labels = batch[2].to(args.device)

                loss, predictions = model(code_inputs, nl_inputs, labels)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                loss_file.write(f'epoch: {cur_epoch}, step {step}, global step: {global_step}, loss: {loss}\n')

                nb_tr_examples += code_inputs.size(0)
                nb_tr_steps += 1
                # loss.backward()
                accelerator.backward(loss)
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

            if args.do_eval:
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                results = evaluate(args, model, tokenizer, accelerator=accelerator)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value, 4))
                    tb_writer.add_scalar(f"eval_{key}", value, cur_epoch)

                eval_acc = results["acc"]
                if eval_acc >= best_results['acc']:
                    not_acc_inc_cnt = 0
                    logger.info("  Best acc: %s", round(eval_acc, 4))
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
                    best_acc = eval_acc
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.data_num == -1 or True:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best acc model into %s", output_model_file)
                else:
                    not_acc_inc_cnt += 1
                    logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
                    if not_acc_inc_cnt > args.patience:
                        logger.info("Early stop as acc do not increase for %d times", not_acc_inc_cnt)
                        fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
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

        for criteria in ['best-acc']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

            results = test(args, model, tokenizer, accelerator)

            for key, value in results.items():
                fa.write(f"  Final test {key} = {str(value)}\n")

            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    f.write("[%s] acc: %.4f\n\n" % (
                        criteria, results['acc']))
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
