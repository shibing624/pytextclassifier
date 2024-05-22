# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    FlaubertModel,
    LongformerModel,
    RemBertModel,
    RemBertPreTrainedModel,
    XLMModel,
    XLMPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
)
from transformers.modeling_utils import SequenceSummary
from transformers.models.albert.modeling_albert import (
    AlbertModel,
    AlbertPreTrainedModel,
)
from transformers.models.longformer.modeling_longformer import (
    LongformerClassificationHead,
    LongformerPreTrainedModel,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
                              2:
                              ]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForHierarchicalMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, pos_weight=None):
        super(BertForHierarchicalMultiLabelSequenceClassification, self).__init__(config)
        config.update({"mlp_size": 1024})
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size + config.num_labels, config.mlp_size),
            nn.ReLU(),
            nn.Linear(config.mlp_size, config.mlp_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(config.mlp_size, config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
            parent_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        concat_output = torch.cat((pooled_output, parent_labels), dim=1)
        mlp_output = self.mlp(concat_output)
        logits = self.classifier(mlp_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RemBertForMultiLabelSequenceClassification(RemBertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(RemBertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.rembert = RemBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.rembert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
                              2:
                              ]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLNetForMultiLabelSequenceClassification(XLNetPreTrainedModel):
    """
    XLNet model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(XLNetForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[
                              1:
                              ]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs


class XLMForMultiLabelSequenceClassification(XLMPreTrainedModel):
    """
    XLM model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(XLMForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[
                              1:
                              ]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs


class AlbertForMultiLabelSequenceClassification(AlbertPreTrainedModel):
    """
    Alber model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(AlbertForMultiLabelSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
                              2:
                              ]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class FlaubertForMultiLabelSequenceClassification(FlaubertModel):
    """
    Flaubert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(FlaubertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = FlaubertModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[
                              1:
                              ]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs


class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
    """
    Longformer model adapted for multilabel sequence classification.
    """

    def __init__(self, config, pos_weight=None):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
    ):
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs
