#!/usr/bin/env bash

me=50

# while read f; do
#     echo "python train/MNet_1000_seg.py -dts $f -me $me &&"
#     # python train/MNet_1000_seg.py -dts $f -me $me 2>&1 | tee -a $0.log
# done < ./config/diagnosis_comparison_list.txt

no_var_lists=(
    ""
    "--no_sig"
    "--no_age"
    "--no_sex"
    "--no_MMSE"
    "--no_age --no_sex --no_MMSE"
    "--no_sig --no_sex --no_MMSE"
    "--no_sig --no_age --no_MMSE"
    "--no_sig --no_age --no_sex"
    )

for no_vars in "${no_var_lists[@]}" ; do    
    while read f; do
        echo "python 2_train_and_eval/MNet_1000_seg.py -dts $f -me $me $no_vars --no_mtl &&"
        python 2_train_and_eval/MNet_1000_seg.py -dts $f -me $me $no_vars --no_mtl 2>&1 | tee -a $0.log
    done < ./config/diagnosis_comparison_list.txt
done
    


# ./2_train_and_eval/sh/MNets_1000_seg_all_combi.sh

