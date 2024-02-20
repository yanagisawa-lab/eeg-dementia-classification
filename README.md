![CI](https://github.com/yanagisawa-lab/eeg-dementia-classification/actions/workflows/MNet_1000_forward.yml/badge.svg)

## EEG dementia classification
Watanabe, Y., Miyazaki, Y., Hata, M., Fukuma, R., Aoki, Y., Kazui, H., Araki, T., Taomoto, D., Satake, Y., Suehiro, T., Sato, S., Kanemoto, H., Yoshiyama, K., Ishii, R., Harada, T., Kishima, H., Ikeda, M., & Yanagisawa, T. (2024). **A deep learning model for the detection of various dementia and MCI pathologies based on resting-state electroencephalography data: A retrospective multicentre study**. *Neural Networks*, 171, 242–250. https://doi.org/10.1016/j.neunet.2023.12.009

## Installation
#### Source code
``` bash
$ git clone git@github.com:yanagisawa-lab/eeg-dementia-classification.git
```
#### Trained model
``` bash
$ pip install eeg_dementia_classification_MNet
```


## Pretrained Weights
Pretrained weights are available on [our Google Drive](https://drive.google.com/file/d/1QZYlEtcd4Szf5K55cNrSxalHcW6UjkaF/view?usp=sharing).
1. Download 'pretrained_weights.tar.gz'.
2. Extract the file using the following command:
``` bash
$ tar xvf pretrained_weights.tar.gz
```
3. Overwrite the './pretrained_weights' directory with the extracted directory. As an illustration, the weight files (.pth) should be organized as follows:
```
./eeg_dementia_classification/pretrained_weights/
├── AD_vs_DLB
│   ├── model_fold#0_epoch#045.pth
│   ├── model_fold#1_epoch#031.pth
│   ├── model_fold#2_epoch#029.pth
│   ├── model_fold#3_epoch#031.pth
│   └── model_fold#4_epoch#028.pth
├── AD_vs_DLB_vs_NPH
│   ├── model_fold#0_epoch#024.pth
│   ├── model_fold#1_epoch#035.pth
...
```

## Usage of the Trained Models
``` python
from eeg_dementia_classification_MNet import MNet_1000
import torch

## Parameters
DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]

## MNet
model = MNet_1000(DISEASE_TYPES, is_ensemble=True).cuda()
model.load_weights(i_fold=0) # the pretrained_weights directory should be located at the current working directory

## Feeds data
bs, n_chs, seq_len = 16, 19, 1000
x = torch.rand(bs, n_chs, seq_len).cuda()
y = model(x)
```

## BibTeX Citation
``` bibtex
@article{WATANABE2024242,
title = {A deep learning model for the detection of various dementia and MCI pathologies based on resting-state electroencephalography data: A retrospective multicentre study},
journal = {Neural Networks},
volume = {171},
pages = {242-250},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2023.12.009},
url = {https://www.sciencedirect.com/science/article/pii/S0893608023007037},
author = {Yusuke Watanabe and Yuki Miyazaki and Masahiro Hata and Ryohei Fukuma and Yasunori Aoki and Hiroaki Kazui and Toshihiko Araki and Daiki Taomoto and Yuto Satake and Takashi Suehiro and Shunsuke Sato and Hideki Kanemoto and Kenji Yoshiyama and Ryouhei Ishii and Tatsuya Harada and Haruhiko Kishima and Manabu Ikeda and Takufumi Yanagisawa},
```
