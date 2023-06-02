import os

import torch
from transformers import RobertaPreTrainedModel
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_outputs import MaskedLMOutput


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
