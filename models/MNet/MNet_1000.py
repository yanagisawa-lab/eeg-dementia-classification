#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import mngs


class MNet_1000(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.activation_fc1 = []
        self.activation_fc2 = []
        self.activation_fc3 = []
        
        # basic
        self.config = config
        self.n_chs = len(self.config["montage"])

        # for var-masking exp
        if config["no_sig"]:
            print("\nSignal will be masked with random values ~ N(0,1).\n")
        if config["no_age"]:
            print("\nAge will be masked with 0.\n")
        if config["no_sex"]:
            print("\nSex will be masked with 0.\n")
        if config["no_MMSE"]:
            print("\nMMSE will be masked with 0.\n")

        # for ch-masking exp
        self.indi_to_mask, self.montages_to_mask = mngs.general.search(
            self.config["montages_to_mask"], self.config["montage"]
        )
        if len(self.montages_to_mask) > 0:
            print(
                f"\nSignals on {self.montages_to_mask} will be masked with random values.\n"
            )

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
            nn.Linear(n_fc_in, config["n_fc1"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc1"]),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc2"]),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], len(config["disease_types"])),
        )

        self.fc_subj = nn.Sequential(
            nn.Linear(n_fc_in, config["n_fc1"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc1"]),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.Mish(),
            # nn.BatchNorm1d(config["n_fc2"]),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], config["n_subjs_tra"]),
        )

        if config.get("save_activations", False):
            self.fc_diag[0].register_forward_hook(self.get_activation_fc1())
            self.fc_diag[3].register_forward_hook(self.get_activation_fc2())
            self.fc_diag[6].register_forward_hook(self.get_activation_fc3())                    
        

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

    def forward(self, x, Ab, Sb, Mb):
        # for var-masking exp.
        if self.config["no_sig"]:
            x = torch.randn_like(x, device=x.device)
        if self.config["no_age"]:
            Ab = torch.zeros_like(Ab, device=Ab.device)
        if self.config["no_sig"]:
            Sb = torch.zeros_like(Sb, device=Sb.device)
        if self.config["no_sig"]:
            Mb = torch.zeros_like(Mb, device=x.device)

        # for ch-masking exp.
        x[:, self.indi_to_mask, :] = torch.randn(
            x[:, self.indi_to_mask, :].shape, device=x.device
        )

        # time-wise normalization
        x = self._normalize_time(x)

        x = self._reshape_input(x, self.n_chs)

        x = self.act1(self.conv1(x))
        x = self.act2(self.pool2(self.bn2(self.conv2(x))))
        x = self.swap(x)
        x = self.act3(self.pool3(self.bn3(self.conv3(x))))
        x = self.act4(self.pool4(self.bn4(self.conv4(x))))
        x = x.reshape(len(x), -1)

        # adds age, sex (gender), and MMSE score
        x = torch.hstack([x, Ab.unsqueeze(-1), Sb.unsqueeze(-1), Mb.unsqueeze(-1)])

        y_diag = self.fc_diag(x)
        y_subj = self.fc_subj(x)

        return y_diag, y_subj

    def get_activation_fc1(self,):
        def hook(model, input, output):
            self.activation_fc1.append(output.detach().cpu().numpy())
        return hook
    
    def get_activation_fc2(self,):
        def hook(model, input, output):
            self.activation_fc2.append(output.detach().cpu().numpy())
        return hook
    
    def get_activation_fc3(self,):
        def hook(model, input, output):
            self.activation_fc3.append(output.detach().cpu().numpy())
        return hook
    

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
    inp = torch.rand(bs, n_chs, seq_len)
    Ab = torch.rand(
        bs,
    )
    Sb = torch.rand(
        bs,
    )
    Mb = torch.rand(
        bs,
    )

    ## Config for the model
    model_config = mngs.general.load("./eeg_dementia_classification/models/MNet/MNet_1000.yaml")
    MONTAGE_19 = [
        "FP1",
        "FP2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
    ]
    montages_to_mask = [
        "FP1",
        "FP2",
    ]

    model_config.update(
        dict(
            disease_types=["HV", "AD", "DLB", "NPH"],
            montage=MONTAGE_19,
            montages_to_mask=montages_to_mask,
            n_subjs_tra=1024,
            no_sig=False,
            no_age=False,
            no_sex=False,
            no_MMSE=False,
            no_fc_subj=True,
            save_activations=True,
        )
    )
    model = MNet_1000(model_config)

    y_diag, y_subj = model(inp, Ab, Sb, Mb)
    summary(model, inp, Ab, Sb, Mb)
    print(y_diag.shape)
