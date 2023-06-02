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
from io import open
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForCausalLM, RobertaTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)
import multiprocessing
import time
import json

from models import CompletionModel
from configs import add_args, set_seed
from utils import get_filenames, get_elapse_time, get_special_tokens
from models import get_model_size
from _utils import CompletionDataset, CompletionEvalDataset

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForCausalLM, RobertaTokenizer),
                 'vanilla': (RobertaConfig, RobertaForCausalLM, RobertaTokenizer),
                 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}

cpu_cont = multiprocessing.cpu_count()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def post_process(args, preds, gts, true_gts, saved_file):
    wf = open(saved_file, "w")

    cnt = 0
    new_gt = []
    new_pred = []
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if gt in ["", "<pad>"]:
            continue
        new_gt.append(gt)
        new_pred.append(pred.replace(" ", ""))
        if gt == "</s>":
            gt_str = " ".join(new_gt)
            pred_str = " ".join(new_pred)
            assert gt_str == true_gts[cnt].strip(), f"{cnt} sample gt_str != true_gt"
            wf.write(pred_str + "\n")
            cnt += 1
            new_gt = []
            new_pred = []

    return cnt


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

    # get special tokens
    lit_file = os.path.join(args.data_dir, args.task, "literals.json")
    special_tokens = get_special_tokens(lit_file)

    # Build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_name_or_path == 'none':
        config = config_class.from_pretrained('roberta-base')
        model = model_class(config)
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=True,
                                                sep_token='<EOL>',
                                                bos_token='<s>',
                                                eos_token='</s>',
                                                pad_token='<pad>',
                                                unk_token='<|UNKNOWN|>',
                                                additional_special_tokens=special_tokens)
    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))

    model = CompletionModel(model, config, tokenizer, args)
    logger.info("Finish loading model [%s] from %s", get_model_size(model), args.model_name_or_path)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    args.block_size = 1024

    pool = multiprocessing.Pool(cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
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
        # train_examples, train_data = load_and_cache_completion_data(args, args.train_filename, pool, tokenizer, 'train',
        #                                                         is_sample=False)

        train_data = CompletionDataset(tokenizer, args, logger, file_type="train", block_size=args.block_size)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

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

        global_step, best_acc = 0, 0
        not_acc_inc_cnt = 0
        is_early_stop = False
        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            last_time = time.time()
            model.train()
            for step, batch in enumerate(bar):
                inputs, labels = (batch, batch)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                outputs = model(inputs)
                loss = outputs[0]

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

            # if args.do_eval:
            #     logger.info("***** CUDA.empty_cache() *****")
            #     torch.cuda.empty_cache()
            #
            #     eval_examples, eval_data = load_and_cache_defect_data(args, args.dev_filename, pool, tokenizer,
            #                                                           'valid', is_sample=False)
            #
            #     result = evaluate(args, model, eval_examples, eval_data)
            #     eval_acc = result['eval_acc']
            #
            #     if args.data_num == -1:
            #         tb_writer.add_scalar('dev_acc', round(eval_acc, 4), cur_epoch)
            #
            #     # save last checkpoint
            #     last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            #     if not os.path.exists(last_output_dir):
            #         os.makedirs(last_output_dir)
            #
            #     if True or args.data_num == -1 and args.save_last_checkpoints:
            #         model_to_save = model.module if hasattr(model, 'module') else model
            #         output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            #         torch.save(model_to_save.state_dict(), output_model_file)
            #         logger.info("Save the last model into %s", output_model_file)
            #
            #     if eval_acc > best_acc:
            #         not_acc_inc_cnt = 0
            #         logger.info("  Best acc: %s", round(eval_acc, 4))
            #         logger.info("  " + "*" * 20)
            #         fa.write("[%d] Best acc changed into %.4f\n" % (cur_epoch, round(eval_acc, 4)))
            #         best_acc = eval_acc
            #         # Save best checkpoint for best ppl
            #         output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
            #         if not os.path.exists(output_dir):
            #             os.makedirs(output_dir)
            #         if args.data_num == -1 or True:
            #             model_to_save = model.module if hasattr(model, 'module') else model
            #             output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            #             torch.save(model_to_save.state_dict(), output_model_file)
            #             logger.info("Save the best ppl model into %s", output_model_file)
            #     else:
            #         not_acc_inc_cnt += 1
            #         logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)
            #         if not_acc_inc_cnt > args.patience:
            #             logger.info("Early stop as acc do not increase for %d times", not_acc_inc_cnt)
            #             fa.write("[%d] Early stop as not_acc_inc_cnt=%d\n" % (cur_epoch, not_acc_inc_cnt))
            #             is_early_stop = True
            #             break
            #
            #     model.train()
            #
            # if is_early_stop:
            #     break

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

        # file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
        # logger.info("Reload model from {}".format(file))
        # model.load_state_dict(torch.load(file))

        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)

        eval_dataset = CompletionEvalDataset(tokenizer, args, logger, file_type="test", block_size=args.block_size)

        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.to(args.device)

        def DecodeIds(idxs):
            codes = ""
            for idx in idxs:
                to_add = tokenizer.convert_ids_to_tokens(idx)
                if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                    if not codes.endswith(" "):
                        codes += " " + to_add[1:]
                    else:
                        codes += to_add[1:]
                elif (
                        idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                tokenizer.pad_token_id] or
                        tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
                ):
                    codes += " " + to_add + " "
                else:
                    codes += to_add
            return codes.strip(" ")

        model.eval()

        correct = 0.0
        total = 0

        total_pred = []
        total_gt = []

        for step, batch in enumerate(eval_dataloader):
            inputs = batch.to(args.device)

            with torch.no_grad():
                outputs = model(inputs)
                pred_scores = outputs[0]
                pred_ids = pred_scores.argmax(-1)

            all_pred = []
            all_gt = []
            prev_pred = None
            for pred, gt in zip(pred_ids, inputs):
                pred = pred.cpu().tolist()
                gt = gt.cpu().tolist()

                for i, y in enumerate(gt):
                    if i == 0:
                        if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                 tokenizer.pad_token_id]:
                            now_gt = [y]
                            now_pred = [0] if prev_pred is None else [prev_pred]
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                        else:
                            now_gt = [y]
                            now_pred = [0] if prev_pred is None else [prev_pred]
                    else:
                        if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                            if len(now_gt) > 0:
                                try:
                                    all_pred.append(DecodeIds(now_pred).strip().split()[0])
                                except IndexError:
                                    all_pred.append("<SPACE>")
                                all_gt.append(DecodeIds(now_gt).strip())
                                now_gt = []
                                now_pred = []
                        if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                 tokenizer.pad_token_id] or tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT"):
                            if len(now_gt) > 0:
                                try:
                                    all_pred.append(DecodeIds(now_pred).strip().split()[0])
                                except IndexError:
                                    all_pred.append("<SPACE>")
                                all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = [y]
                            now_pred = [pred[i - 1]]
                            try:
                                all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                            continue
                        now_gt.append(y)
                        now_pred.append(pred[i - 1])
            assert len(all_pred) == len(all_gt)

            total_pred.extend(all_pred)
            total_gt.extend(all_gt)

            for x, y in zip(all_pred, all_gt):
                if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                    total += 1
                    if x == y:
                        correct += 1

        # pickle.dump(total_pred, open(os.path.join(args.output_dir, "preds.pkl"), "wb"))
        # pickle.dump(total_gt, open(os.path.join(args.output_dir, "gts.pkl"), "wb"))

        saved_file = os.path.join(args.output_dir, "predictions.txt")
        total_samples = post_process(args, total_pred, total_gt,
                                     open(args.test_filename).readlines(), saved_file)
        logger.info(f"Eval on {total_samples}, saved at {saved_file}")

        logger.info(f"Test total tokens: {total}, accuracy: {correct / total}")

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
