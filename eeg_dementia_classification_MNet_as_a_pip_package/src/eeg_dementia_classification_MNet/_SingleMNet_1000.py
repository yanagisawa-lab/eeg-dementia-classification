#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from eeg_dementia_classification_MNet.utils import connect_strs
from glob import glob


class SingleMNet_1000(nn.Module):
    def __init__(self, disease_types):

        n_fc1 = 1024
        d_ratio1 = 0.85
        n_fc2 = 256
        d_ratio2 = 0.85
        window_size_pts = 1000
        n_subjs_tra = 1024

        super().__init__()

        # basic
        self.disease_types = disease_types
        self.n_chs = 19

        # conv
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(19, 4))
        self.act1 = nn.Mish()

        self.conv2 = nn.Conv2d(40, 40, kernel_size=(1, 4))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d((1, 5))
        self.act2 = nn.Mish()

        self.swap = SwapLayer()

        self.conv3 = nn.Conv2d(1, 50, kernel_size=(8, 12))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d((3, 3))
        self.act3 = nn.Mish()

        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 5))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.act4 = nn.Mish()

        # fc
        n_fc_in = 15950 + 3

        self.fc_diag = nn.Sequential(
            nn.Linear(n_fc_in, n_fc1),
            nn.Mish(),
            # nn.BatchNorm1d(n_fc1),
            nn.Dropout(d_ratio1),
            nn.Linear(n_fc1, n_fc2),
            nn.Mish(),
            # nn.BatchNorm1d(n_fc2),
            nn.Dropout(d_ratio2),
            nn.Linear(n_fc2, len(disease_types)),
        )

    @staticmethod
    def _reshape_input(x, n_chs):
        """
        (batch, channel, time_length) -> (batch, channel, time_length, new_axis)
        """
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.shape[2] == n_chs:
            x = x.transpose(1, 2)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    @staticmethod
    def _normalize_time(x):
        return (x - x.mean(dim=-1, keepdims=True)) / x.std(dim=-1, keepdims=True)

    def load_weights(self, i_fold=0):
        weights_dir = (
            f"./pretrained_weights/{connect_strs(self.disease_types, filler='_vs_')}/"
        )
        weights_list = glob(weights_dir + f"model_fold#{i_fold}_epoch#*.pth")
        if not weights_list:  # This checks if the list is empty
            error_message = (
                "Pretrained weights are not found.\n"
                "1. Download them from our Google Drive (https://drive.google.com/file/d/1QZYlEtcd4Szf5K55cNrSxalHcW6UjkaF/view)\n"
                "2. Extract the pretrained_weights.tar.gz with 'tar xvf pretrained_weights.tar.gz'\n"
                "3. Locate the pretrained_weights directory in the current working directory."
            )
            raise ValueError(error_message)
        weights_path = weights_list[0]
        weights = torch.load(weights_path)
        # Remove keys related to fc_subj
        weights = {k: v for k, v in weights.items() if not k.startswith("fc_subj")}
        self.load_state_dict(weights, strict=False)
        print(f"\nPretrained weights have been loaded from {weights_path}.\n")
        return self

    def forward(self, x):
        x = self._normalize_time(x)  # time-wise normalization
        x = self._reshape_input(
            x, self.n_chs
        )  # x.shape: torch.Size([16, 1, 19, 1000]): bs, 1, n_chs, seq_len

        x = self.act1(self.conv1(x))
        x = self.act2(self.pool2(self.bn2(self.conv2(x))))
        x = self.swap(x)
        x = self.act3(self.pool3(self.bn3(self.conv3(x))))
        x = self.act4(self.pool4(self.bn4(self.conv4(x))))
        x = x.reshape(len(x), -1)

        # for Age, Sex, and MMSE score
        x = torch.cat([x, torch.zeros([len(x), 3], device=x.device)], dim=1)
        y_diag = self.fc_diag(x)

        return y_diag


class SwapLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)


if __name__ == "__main__":
    ## Demo data
    bs, n_chs, seq_len = 16, 19, 1000
    x = torch.rand(bs, n_chs, seq_len)

    disease_types = ["HV", "AD", "DLB", "NPH"]
    model = SingleMNet_1000(disease_types)

    y_diag = model(x)
    # summary(model, x)

    model.load_weights(i_fold=0)
