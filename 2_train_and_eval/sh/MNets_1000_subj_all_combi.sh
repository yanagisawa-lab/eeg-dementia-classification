#!/usr/bin/env bash

#me=50

while read f; do
            echo "python 2_train_and_evalMNet_1000_subj.py -dts $f &&"
            python 2_train_and_evalMNet_1000_subj.py -dts $f 2>&1 | tee -a $0.log
done < ./config/diagnosis_comparison_list.txt

for dts in 'HV AD DLB NPH'; do
    echo "python 2_train_and_evalMNet_1000_subj.py -dts $f &&"
    python 2_train_and_evalMNet_1000_subj.py -dts $f 2>&1 | tee -a $0.log
done

f="AD DLB NPH"
echo "python 2_train_and_evalMNet_1000_subj.py -dts $f &&"
python 2_train_and_evalMNet_1000_subj.py -dts $f 2>&1 | tee -a $0.log


while read f; do
    echo "python 2_train_and_evalMNet_1000_subj.py -dts $f &&"
    python 2_train_and_evalMNet_1000_subj.py -dts $f 2>&1 | tee -a $0.log
done < ./config/diagnosis_comparison_list.txt


while read f; do
    echo "python 2_train_and_evalMNet_1000_subj.py -dts $f &&"
    python 2_train_and_evalMNet_1000_subj.py -dts $f 2>&1 | tee -a $0.log
done < ./config/diagnosis_comparison_list.txt

while read f; do
    echo "python 2_train_and_evalMNet_1000_subj.py -dts $f &&"
    python 2_train_and_evalMNet_1000_subj.py -dts $f 2>&1 | tee -a $0.log
done < ./config/diagnosis_comparison_list.txt

# ./2_train_and_evalsh/MNets_1000_subj_all_combi.sh

