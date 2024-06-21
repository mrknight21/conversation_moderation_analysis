import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.longformer.configuration_longformer import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerSequenceClassifierOutput, LongformerModel,\
    LongformerPreTrainedModel, LONGFORMER_START_DOCSTRING, LONGFORMER_INPUTS_DOCSTRING
from transformers.models.led import LEDModel, LEDConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LongformerConfig"

@dataclass
class LongformerSequenceClassifierMultitasksOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x +
            attention_window + 1)`, where `x` is the number of tokens with global attention mask.

            Local attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token in the sequence to every token with
            global attention (first `x` values) and to every token in the attention window (remaining `attention_window
            + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the
            remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a
            token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding
            (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens.
            If the attention window contains a token with global attention, the attention weight at the corresponding
            index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global
            attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be
            accessed from `global_attentions`.
        global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
            where `x` is the number of tokens with global attention mask.

            Global attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads. Those are the attention weights from every token with global attention to every token
            in the sequence.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: dict[str, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@add_start_docstrings(
    """
    Longformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    LONGFORMER_START_DOCSTRING,
)
class LongformerForSequenceMultiTasksClassification(LongformerPreTrainedModel):
    def __init__(self, attributes_info, longformer_encoder_base="allenai/longformer-base-4096", checkpoint=None):
        if "led" in longformer_encoder_base.lower():
            config = LEDConfig.from_pretrained(longformer_encoder_base)
        else:
            config = LongformerConfig.from_pretrained(longformer_encoder_base)

        super().__init__(config)
        if not checkpoint:
            if "led" in longformer_encoder_base.lower():
                led = LEDModel.from_pretrained(longformer_encoder_base)
                self.longformer = led.encoder
            else:
                self.longformer = LongformerModel.from_pretrained(longformer_encoder_base)

            self.tasks_info = attributes_info
            self.classifiers = nn.ModuleDict()
            self.attributes_count = len(attributes_info)
            self.attributes_total_labels_count = 0

            for att, info in attributes_info.items():

                self.classifiers[att] = LongformerClassificationHead(config, info['num_labels'])

            # Initialize weights and apply final processing
            self.post_init()

    @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            logger.warning_once("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]
        all_logits = torch.empty(batch_size, 0).to(sequence_output.device)
        loss = torch.tensor(0.0, dtype=torch.float).to(sequence_output.device)
        task_index = 0
        for k, info in self.tasks_info.items():
            clsr = self.classifiers[k]
            num_labels = self.tasks_info[k]['num_labels']
            logits = clsr(sequence_output)
            task_labels = labels[:, task_index]
            task_labels = task_labels.to(logits.device)

            if self.config.problem_type is None:
                if num_labels == 1:
                    self.config.problem_type = "regression"
                elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if num_labels == 1:
                    loss += loss_fct(logits.squeeze(), task_labels.squeeze())
                else:
                    loss += loss_fct(logits, task_labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(logits.view(-1, num_labels), task_labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss += loss_fct(logits, task_labels)

            all_logits = torch.cat((all_logits, logits), dim=1)
            task_index += 1



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierMultitasksOutput(
            loss=loss,
            logits=all_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )


class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output