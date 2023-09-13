#!/usr/bin/env python3
# Time-stamp: "2021-12-21 21:41:38 (ywatanabe)"

import torch
import torch.nn as nn
import mngs

from functools import partial


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        samp_rate,
        features_list=[
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "median",
            "q25",
            "q75",
            "rms",
            "rfft_bands",
            "beyond_r_sigma_ratio",
        ],
        batch_size=None,
    ):
        super().__init__()

        self.func_dict = dict(
            mean=mngs.dsp.mean,
            std=mngs.dsp.std,
            skewness=mngs.dsp.skewness,
            kurtosis=mngs.dsp.kurtosis,
            median=mngs.dsp.median,
            q25=mngs.dsp.q25,
            q75=mngs.dsp.q75,
            rms=mngs.dsp.rms,
            rfft_bands=partial(mngs.dsp.rfft_bands, samp_rate=samp_rate),
            beyond_r_sigma_ratio=mngs.dsp.beyond_r_sigma_ratio,
        )

        self.features_list = features_list

        self.batch_size = batch_size

    def forward(self, x):
        if self.batch_size is None:
            conc = torch.cat(
                [self.func_dict[f_str](x) for f_str in self.features_list], dim=-1
            )
            return conc
        else:
            conc = []
            n_batches = len(x) // self.batch_size + 1
            for i_batch in range(n_batches):
                try:
                    start = i_batch * self.batch_size
                    end = (i_batch + 1) * self.batch_size
                    conc.append(
                        torch.cat(
                            [
                                self.func_dict[f_str](x[start:end].cuda())
                                for f_str in self.features_list
                            ],
                            dim=-1,
                        ).cpu()
                    )
                except Exception as e:
                    print(e)
            return torch.cat(conc)


def main():
    BS = 32
    N_CHS = 19
    SEQ_LEN = 2000
    SAMP_RATE = 1000
    NYQ = int(SAMP_RATE / 2)

    x = torch.randn(BS, N_CHS, SEQ_LEN).cuda()

    m = FeatureExtractor(SAMP_RATE, batch_size=8)

    out = m(x)  # 15 features


if __name__ == "__main__":
    main()
