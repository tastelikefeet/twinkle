# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
from twinkle.data_format import LossOutput


class MSELoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        import torch
        preds = outputs['logits']
        labels = inputs['labels']
        return LossOutput(loss=torch.nn.MSELoss()(preds, labels))
