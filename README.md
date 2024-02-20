![CI](https://github.com/yanagisawa-lab/eeg-dementia-classification/actions/workflows/MNet_1000_forward.yml/badge.svg)

## EEG_dementia_classification
This repository contains the source code used in the study titled "[A Deep Learning Model for Detection of Dementia Diseases and Identification of Underlying Pathologies of Mild Cognitive Impairment Based on Resting-State Electroencephalography: A Retrospective, Multicenter Study](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4304749)"

``` bash
git clone git@github.com:yanagisawa-lab/eeg-dementia-classification.git
```

## Pretrained Weights
Pretrained weights are available on [our Google Drive](https://drive.google.com/file/d/1QZYlEtcd4Szf5K55cNrSxalHcW6UjkaF/view?usp=sharing).
1. Download 'pretrained_weights.tar.gz'.
2. Extract the file using the following command:
``` bash
$ tar xvf pretrained_weights.tar.gz
```
3. Overwrite the './data/pretrained_weights' directory with the extracted directory. As an illustration, the weight files (.pth) should be organized as follows:
```
./eeg_dementia_classification/data/pretrained_weights/
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

## Trained model installation
``` bash
pip install eeg_dementia_classification_MNet
```

## Usage
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
