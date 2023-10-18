## Installation
``` bash
$ pip install eeg-dementia-classification-MNet
```

## Pretrained Weights
Pretrained weights are available on [our Google Drive](https://drive.google.com/file/d/1QZYlEtcd4Szf5K55cNrSxalHcW6UjkaF/view?usp=sharing).
1. Download 'pretrained_weights.tar.gz'.
2. Extract the file using the following command:
``` bash
$ tar xvf pretrained_weights.tar.gz
```
3. Locate the extradcted 'pretrained_weights' directory at the working directory. As an illustration, the weight files (.pth) should be organized as follows:
```
./pretrained_weights/
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

## Usage
``` python
from eeg_dementia_classification_MNet import MNet_1000
import torch

## Parameters
DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]

## MNet
model = MNet_1000(DISEASE_TYPES, is_ensemble=True)
model.load_weights(i_fold=0)

## Feeds data
bs, n_chs, seq_len = 16, 19, 1000
x = torch.rand(bs, n_chs, seq_len)
y = model(x)
```

## Contact
Please feel free to contact [the author](mailto:ywata1989@gmail.com).
