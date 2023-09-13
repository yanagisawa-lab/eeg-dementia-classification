#!/usr/bin/env python3

import mngs
import numpy as np
import pandas as pd


################################################################################
## Pandas
################################################################################
def merge_columns(_df, *columns):
    """
    Add merged columns in string.

    DF = pd.DataFrame(data=np.arange(25).reshape(5,5),
                      columns=["A", "B", "C", "D", "E"],
    )

    print(DF)

    # A   B   C   D   E
    # 0   0   1   2   3   4
    # 1   5   6   7   8   9
    # 2  10  11  12  13  14
    # 3  15  16  17  18  19
    # 4  20  21  22  23  24

    print(merge_columns(DF, "A", "B", "C"))

    #     A   B   C   D   E     A_B_C
    # 0   0   1   2   3   4     0_1_2
    # 1   5   6   7   8   9     5_6_7
    # 2  10  11  12  13  14  10_11_12
    # 3  15  16  17  18  19  15_16_17
    # 4  20  21  22  23  24  20_21_22
    """
    from copy import deepcopy

    df = deepcopy(_df)
    merged = deepcopy(df[columns[0]])  # initialization
    for c in columns[1:]:
        merged = mngs.ml.utils.merge_labels(list(merged), deepcopy(df[c]))
    df.loc[:, mngs.general.connect_strs(columns)] = merged
    return df


def col_to_last(df, col):
    df_orig = df.copy()
    cols = list(df_orig.columns)
    cols_remain = pop_keys_from_keys_list(cols, col)
    out = pd.concat((df_orig[cols_remain], df_orig[col]), axis=1)
    return out


def col_to_top(df, col):
    df_orig = df.copy()
    cols = list(df_orig.columns)
    cols_remain = pop_keys_from_keys_list(cols, col)
    out = pd.concat((df_orig[col], df_orig[cols_remain]), axis=1)
    return out


# class IDAllocator(object):
#     """
#     Example1:
#         patterns = np.array([3, 2, 1, 2, 50, 20])
#         alc = IDAllocator(patterns)
#         input = np.array([2, 20, 3, 1])
#         IDs = alc(input)
#         print(IDs) # [1, 3, 2, 0]

#     Example2:
#         patterns = np.array(['a', 'b', 'c', 'zzz'])
#         alc = IDAllocator(patterns)
#         input = np.array(['c', 'a', 'zzz', 'b'])
#         IDs = alc(input)
#         print(IDs) # [2, 0, 3, 1]
#     """

#     def __init__(self, patterns):
#         patterns_uq = np.unique(patterns)  # natural sorting is executed.
#         new_IDs = np.arange(len(patterns_uq))
#         self.correspondence_table = pd.DataFrame(
#             {
#                 "Original": patterns_uq,
#                 "new_ID": new_IDs,
#             }
#         ).set_index("Original")

#     def __call__(self, x):
#         allocated = np.array(self.correspondence_table.loc[x]).squeeze()
#         return allocated


class IDAllocator(object):
    """
    Example1:
        # registers patterns
        patterns = np.array([3, 2, 1, 2, 50, 20])
        alc = IDAllocator(patterns)

        # orig to new
        input = np.array([2, 20, 3, 1])
        IDs = alc(input) # alc.to_new(input)
        print(IDs) # [1, 3, 2, 0]

        # new to orig
        print(alc.to_orig(IDs)) # array([ 2, 20,  3,  1])
        print(alc.table)

    Example2:
        # registers patterns
        patterns = np.array(['a', 'b', 'c', 'zzz'])
        alc = IDAllocator(patterns)
        print(alc.table)

        # orig to new
        input = np.array(['c', 'a', 'zzz', 'b'])
        IDs = alc(input) # alc.to_new(input)
        print(IDs) # [2, 0, 3, 1]

        # new to orig
        reversed = alc.to_orig(IDs)
        print(reversed) # array(['c', 'a', 'zzz', 'b'])
    """

    def __init__(self, patterns):

        orig_uq = np.unique(patterns)  # natural sorted
        new_IDs = np.arange(len(orig_uq))

        self.table = pd.DataFrame(
            {
                "Original": orig_uq,
                "new_ID": new_IDs,
            }
        ).set_index("Original")

        self.correspondence_table = self.table  # alias for the backward compatibility

        self.orig_to_new_dict = {o: n for o, n in zip(orig_uq, new_IDs)}

        self.new_to_orig_dict = {v: k for k, v in self.orig_to_new_dict.items()}

    def __call__(self, x):  # alias for self.to_new(x)
        return self.to_new(x)

    def to_new(self, xo):
        return np.array([self.orig_to_new_dict[xo_i] for xo_i in xo])

    def to_orig(self, xn):
        return np.array([self.new_to_orig_dict[xn_i] for xn_i in xn])


def force_dataframe(permutable_dict, filler=""):
    ## Deep copy
    permutable_dict = permutable_dict.copy()

    ## Get the lengths
    max_len = 0
    for k, v in permutable_dict.items():
        max_len = max(max_len, len(v))

    ## Replace with 0 filled list
    for k, v in permutable_dict.items():
        permutable_dict[k] = list(v) + (max_len - len(v)) * [filler]

    ## Puts them into a DataFrame
    out_df = pd.DataFrame(permutable_dict)

    return out_df
