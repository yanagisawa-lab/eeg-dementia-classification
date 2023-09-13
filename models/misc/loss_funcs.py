#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mngs


class MultiTaskLoss(nn.Module):
    """
    # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf

    Example:
        are_regression = [False, False]
        mtl = MultiTaskLoss(are_regression)
        losses = [torch.rand(1, requires_grad=True) for _ in range(len(are_regression))]
        loss = mtl(losses)
        print(loss)
        # [tensor([0.4215], grad_fn=<AddBackward0>), tensor([0.6190], grad_fn=<AddBackward0>)]
    """

    def __init__(self, are_regression=[False, False], reduction="none"):
        super().__init__()
        mngs.general.fix_seeds(np=np, torch=torch, show=False)
        n_tasks = len(are_regression)

        self.register_buffer("are_regression", torch.tensor(are_regression))

        # for the numercal stability, log(variables) are learned.
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        # losses = torch.Tensor(losses).type_as(losses[0][0])

        vars = torch.exp(self.log_vars).type_as(losses[0])
        stds = vars ** (1 / 2)

        coeffs = 1 / ((self.are_regression + 1) * vars)

        scaled_losses = [
            coeffs[i] * losses[i] + torch.log(stds[i]) for i in range(len(losses))
        ]

        return scaled_losses


def L1_reguralization(model, lambda_l1=0.01):
    lambda_l1 = torch.tensor(lambda_l1)
    l1 = torch.tensor(0.0).cuda()
    for param in model.parameters():
        l1 += torch.abs(param).sum()
    return l1


def L2_reguralization(model, lambda_l2=0.01):
    lambda_l2 = torch.tensor(lambda_l2)
    l2 = torch.tensor(0.0).cuda()
    for param in model.parameters():
        l2 += torch.norm(param).sum()
    return l2


def add_L2_loss(loss, model, lambda_l2=0.01):
    lambda_l2 = torch.tensor(lambda_l2).type_as(loss)
    l2 = torch.tensor(0.0).type_as(loss)
    for param in model.parameters():
        l2 += torch.norm(param).sum()
    return loss + l2


def Elastic_Reguralization(model, alpha=1.0, l1_ratio=0.5):
    # a * L1 + b * L2
    # alpha = a + b
    # l1_ratio = a / (a + b)
    L1 = L1_reguralization(model)
    L2 = L2_reguralization(model)
    a = alpha * l1_ratio
    b = alpha * (1 - l1_ratio)
    elastic_reguralization = a * L1 + b * L2
    return elastic_reguralization


def CosineLoss(x, t, **kwargs):
    _t = t * (2 - 1)  # 0, 1 -> -1, 1
    return nn.CosineEmbeddingLoss(x, _t, **kwargs)


# import tensorflow as tf
# from tensorflow.keras import backend as K

# def multi_mcc_loss(y_true, y_pred, false_pos_penal=1.0):
#     # https://github.com/vlainic/matthews-correlation-coefficient/blob/master/multi_mcc_loss.py
#     confusion_m = tf.matmul(K.transpose(y_true), y_pred)
#     if false_pos_penal != 1.0:
#       """
#       This part is done for penalization of FalsePos symmetrically with FalseNeg,
#       i.e. FalseNeg is favorized for the same factor. In such way MCC values are comparable.
#       If you want to penalize FalseNeg, than just set false_pos_penal < 1.0 ;)
#       """
#       confusion_m = tf.matrix_band_part(confusion_m, 0, 0) \
#                   + tf.matrix_band_part(confusion_m, 0, -1)*false_pos_penal \
#                   + tf.matrix_band_part(confusion_m, -1, 0)/false_pos_penal

#     N = K.sum(confusion_m)

#     up = N*tf.trace(confusion_m) - K.sum(tf.matmul(confusion_m, confusion_m))
#     down_left = K.sqrt(N**2 - K.sum(tf.matmul(confusion_m, K.transpose(confusion_m))))
#     down_right = K.sqrt(N**2 - K.sum(tf.matmul(K.transpose(confusion_m), confusion_m)))

#     mcc = up / (down_left * down_right + K.epsilon())
#     mcc = tf.where(tf.is_nan(mcc), tf.zeros_like(mcc), mcc)

#     return 1 - K.mean(mcc)
