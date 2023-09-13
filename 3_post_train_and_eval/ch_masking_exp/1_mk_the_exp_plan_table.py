#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:44:49 (ywatanabe)"

from itertools import combinations, product

import mngs
import numpy as np
import pandas as pd
from tqdm import tqdm

montages = mngs.general.load("./config/load_params.yaml")["montage"]
CH_ALL = [f"{bi[0]}-{bi[1]}" for bi in montages]

count = 2**len(CH_ALL)
index = np.arange(count)
df = pd.DataFrame(columns=CH_ALL,
                  index=index,
                  )

count = 0
for n_picks in tqdm(range(len(CH_ALL) + 1)):
    # print(n_picks)
    combi = list(combinations(CH_ALL, n_picks))
    for i_combi in combi:
        df.loc[count, i_combi] = False
        count += 1
        
df = df.fillna(True)

mngs.general.fix_seeds(np=np)
df["rand_index"] = np.random.permutation(df.index)
df = df.sort_values(["rand_index"])

mngs.io.save(df, "./results/montages_to_mask.csv")

# print(df.iloc[0])
