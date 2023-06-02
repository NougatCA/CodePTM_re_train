import os

import torch
from transformers import RobertaConfig, RobertaForMaskedLM, get_linear_schedule_with_warmup, RobertaPreTrainedModel, \
    RobertaModel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformers.activations import get_activation
from transformers.file_utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import MaskedLMOutput

import config
import utils
from data import MLMDataset, RTDDataset
from utils import logger
from apex import amp


class MLMTrainer:
    def __init__(self, model=None, optimizer=optim.Adam, lr=config.LR,
                 data=os.path.join(config.PATH_TRAIN_DATA, 'MLM_data.json'), vocab=config.PATH_VOCAB):
        logger.info('=' * 100)

        logger.info('Data loading.')
        dataset = MLMDataset(data, vocab)
        self.data_loader = Data.DataLoader(dataset, config.BATCH_SIZE, True)
        logger.info('Data loaded, dataset size: {}'.format(len(self.data_loader)))

        logger.info('Model setting.')
        if model is None:
            self.config = RobertaConfig.from_json_file(config.PATH_MODEL_CONFIG)
            model = RobertaForMaskedLM(self.config).to('cuda')
            logger.info('model.vocab_size: {}'.format(model.config.vocab_size))
        else:
            model = model.to('cuda')
        # self.model = model
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.model, self.optimizer = amp.initialize(model, self.optimizer, opt_level="O1")
        self.model = nn.DataParallel(model, device_ids=[i for i in range(config.NUM_WORKERS)])
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.WARMUP_STEPS,
                                                            num_training_steps=config.TOTAL_STEPS)
        n_parameters = sum(p.numel() for p in self.model.parameters())
        n_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info('Model set. Amount of parameters: {}, including {} trainable parameters.'.format(n_parameters,
                                                                                                     n_trainable_parameters))

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
            for batch, (input_ids, token_type_ids, attention_mask, labels) in enumerate(
                    self.data_loader):
                output = self.model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                loss = output['loss'].mean() / accumulation_steps
                # loss.backward()
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

        logger.info(
            'Training finished. Average time consumed per {} steps: {}'.format(config.TIME_SAVE_STEP, timer.avgtime()))

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
        model = RobertaForMaskedLM(RobertaConfig.from_json_file(config.PATH_MODEL_CONFIG))
        # model = model.module
        # model = nn.DataParallel(model, device_ids=[i for i in range(config.NUM_WORKERS)])
        model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, config.PATH_SAVE_MODEL_MLMModel)))
        return model


class RobertaDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class RobertaForPreTraining(RobertaPreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)

        self.roberta = model
        self.discriminator_predictions = RobertaDiscriminatorPredictions(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see `input_ids` docstring)
            Indices should be in `[0, 1]`:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        Returns:
        Examples:
        ```python
        >>> from transformers import ElectraTokenizer, ElectraForPreTraining
        >>> import torch
        >>> tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
        >>> model = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
        ...     0
        >>> )  # Batch size 1
        >>> logits = model(input_ids).logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class RTDTrainer:
    def __init__(self, generator_model, optimizer=optim.Adam, lr=config.LR,
                 data=os.path.join(config.PATH_TRAIN_DATA, 'RTD_data.json'), vocab=config.PATH_VOCAB):
        logger.info('=' * 100)

        logger.info('Model Setting.')
        self.config = RobertaConfig.from_json_file(config.PATH_MODEL_CONFIG)

        self.model = RobertaForPreTraining(self.config, generator_model.base_model).cuda()
        logger.info('model.vocab_size: {}'.format(self.model.config.vocab_size))
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(config.NUM_WORKERS)])
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.WARMUP_STEPS,
                                                            num_training_steps=config.TOTAL_STEPS)
        n_parameters = sum(p.numel() for p in self.model.parameters())
        n_trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info('Model set. Amount of parameters: {}, including {} trainable parameters.'.format(n_parameters,
                                                                                                     n_trainable_parameters))

        logger.info('Model Set.')

        logger.info('Data loading.')
        dataset = RTDDataset()
        self.data_loader = Data.DataLoader(dataset, config.BATCH_SIZE, True)
        logger.info('Data loaded, dataset size: {}'.format(len(self.data_loader)))

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
            for batch, (input_ids, attention_mask, labels) in enumerate(self.data_loader):
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                loss = output['loss'].mean() / accumulation_steps
                # loss.backward()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if ((batch + 1) % accumulation_steps) is 0:
                    logger.info('Step: {}/{}, Loss: {:.4f}'.format(step, config.TOTAL_STEPS, loss.item()))
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

        logger.info(
            'Training finished. Average time consumed per {} steps: {}'.format(config.TIME_SAVE_STEP, timer.avgtime()))

    def save_model(self, step, path=config.MODEL_SAVE_PATH):
        if not os.path.exists(path):
            os.mkdir(path)
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(),
                       os.path.join(path, '{}_model_parallel_{}_step_{}.ckpt'.format('RTD',
                                                                                     config.NUM_WORKERS, step)))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, '{}_model_step_{}.ckpt'.format('RTD', step)))
