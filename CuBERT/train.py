import os

import torch
from transformers import BertConfig, BertForPreTraining, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import config
import utils
from data import CodeDataset
from utils import logger
from apex import amp

class BertModel(nn.Module):
    def __init__(self, config_file='bert_config.json'):
        super(BertModel, self).__init__()
        self.config = BertConfig.from_json_file(config_file)
        self.model = BertForPreTraining(self.config)

    # @torchsnooper.snoop()
    def forward(self, input_ids, token_type_ids, attention_mask, labels, next_sentence_label):
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            labels=labels, next_sentence_label=next_sentence_label)
        return output


class Trainer:
    def __init__(self, model=None, optimizer=optim.Adam, lr=config.LR,
                 data='pretraining_data/sentence_and_next_label.json', vocab='pretraining_data/vocab.txt'):
        logger.info('=' * 100)

        logger.info('Data loading.')
        dataset = CodeDataset(data, vocab)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # self.data_loader = Data.DataLoader(dataset, config.BATCH_SIZE, True, num_workers=config.NUM_WORKERS,
        #                                    pin_memory=True, drop_last=True, sampler=train_sampler)
        self.data_loader = Data.DataLoader(dataset, config.BATCH_SIZE, False)
        logger.info('Data loaded, dataset size: {}'.format(len(self.data_loader)))

        logger.info('Model setting.')
        # torch.distributed.init_process_group(backend="nccl")
        if model is None:
            model = BertModel().to('cuda')
        else:
            model = model.to('cuda')

        # self.model = torch.nn.parallel.DistributedDataParallel(model)
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.model, self.optimizer = amp.initialize(model, self.optimizer, opt_level="O1")
        self.model = nn.DataParallel(model, device_ids=[i for i in range(config.NUM_WORKERS)])
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.WARMUP_STEPS,
                                                            num_training_steps=config.TOTAL_STEPS)
        n_parameters = sum(p.numel() for p in self.model.parameters())
        n_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info('Model set. Amount of parameters: {}, including {} trainable parameters.'.format(n_parameters, n_trainable_parameters))

        logger.info('=' * 100)

    def run(self):
        logger.info('=' * 100)
        logger.info('Training start.')
        # n_epoch = int(config.TOTAL_STEPS / len(self.data_loader))
        # logger.info('total {} epoch'.format(n_epoch))
        logger.info('total {} step'.format(config.TOTAL_STEPS))
        accumulation_steps = int(config.TARGET_BATCH_SIZE / config.BATCH_SIZE)
        step = 0
        timer = utils.Timer()
        while step < config.TOTAL_STEPS:
            for batch, (input_ids, token_type_ids, attention_mask, labels, is_next_label) in enumerate(
                    self.data_loader):
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels, next_sentence_label=is_next_label)
                loss = output['loss'].mean() / accumulation_steps
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if ((batch + 1) % accumulation_steps) is 0:
                    logger.info(
                        'Step: {}/{}, Loss:   {:.4f}'.format(int((batch + 1) / accumulation_steps), config.TOTAL_STEPS,
                                                             loss.item()))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    step += 1
                    if step % config.TIME_SAVE_STEP == 0:
                        logger.info('The latest {} steps consume {:.2f}s'.format(config.TIME_SAVE_STEP, timer.record()))
                    if step % config.SAVE_STEP == 0:
                        self.save_model(step)

                    if step == config.TOTAL_STEPS:
                        break
            if step == config.TOTAL_STEPS:
                break

        logger.info('Training finished. Average time consumed per {} steps: {}'.format(config.TIME_SAVE_STEP, timer.avgtime()))

    def save_model(self, step, path=config.MODEL_SAVE_PATH):
        if not os.path.exists(path):
            os.mkdir(path)
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), os.path.join(path, 'model_parallel_{}_step_{}.ckpt'.format(
                config.NUM_WORKERS, step)))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, 'model_step_{}.ckpt'.format(step)))

    @classmethod
    def load_model(cls, path=config.MODEL_SAVE_PATH):
        model = BertModel()
        model.load_state_dict(torch.load(path))
        return model
