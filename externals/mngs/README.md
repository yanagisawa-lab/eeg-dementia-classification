## Requirements
```
chardet
GitPython
h5py
joblib
matplotlib
natsort
numpy
pandas
pymatreader
PyYAML
scipy
seaborn
sklearn
statsmodels
torch
xmltodict
```

## Installation
``` bash
$ pip install mngs

or

$ pip install git+https://github.com/ywatanabe1989/mngs.git@develop
```



## mngs.general.save
``` python
import mngs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## numpy
arr = np.arange(10)
mngs.general.save(arr, 'spath.npy')

## pandas
df = pd.DataFrame(arr)
mngs.general.save(df, 'spath.csv')

## matplotlib
fig, ax = plt.subplots()
ax.plot(arr)
mngs.general.save(fig, 'spath.png)
```

## mngs.general.load
``` python
import mngs
arr = mngs.general.load('spath.npy')
arr = mngs.general.load('spath.mat')
df = mngs.general.load('spath.npy')
yaml_dict = mngs.general.load('spath.yaml')
hdf5_dict = mngs.general.load('spath.hdf5')
```

## mngs.general.fix_seeds

``` python
import mngs
import os
import random
import numpy as np
import torch

mngs.general.fix_seeds(os=os, random=random, np=np, torch=torch, tf=None, seed=42)
```

## mngs.general.tee
``` python
import sys
sys.stdout, sys.stderr = tee(sys)
print("abc")  # also wrriten in stdout
print(1 / 0)  # also wrriten in stderr
```

## mngs.plt.configure_mpl
``` python
configure_mpl(
    plt,
    dpi=100,
    figsize=(16.2, 10),
    figscale=1.0,
    fontsize=16,
    labelsize="same",
    legendfontsize="xx-small",
    tick_size="auto",
    tick_width="auto",
    hide_spines=False,
)
```

## mngs.plt.ax_*
- mngs.plt.ax_extend
- mngs.plt.ax_scientific_notation
- mngs.plt.ax_set_position

## mngs.ml.Reporter
Now, classification task is available.
``` python
reporter = mngs.ml.Reporter(sdir=log_dir)
for i_fold in range(N_FOLDS):
    ...
    print("\n--- Metrics ---\n")
    reporter.calc_metrics(
        T_tes,
        pred_class_tes,
        pred_proba_tes,
        labels=[class_0, class_1, class_2],
        i_fold=i_fold,
    )
    print("\n---------------\n")

reporter.summarize()
reporter.save()
```

The above lines makes reportes and figures.
``` bash
$ tree $log_dir
├── aucs.csv
├── bACCs.csv
├── balanced_accs.csv
├── clf_reports.csv
├── conf_mat
│   ├── conf_mats.csv
│   ├── fold#0.png
│   ├── fold#1.png
│   ├── fold#2.png
│   └── overall_sum.png
├── mccs.csv
├── pre_rec_curves
│   ├── fold#0.png
│   ├── fold#1.png
│   └── fold#2.png
└── roc_curves
    ├── fold#0.png
    ├── fold#1.png
    └── fold#2.png
```
