#!/usr/bin/env bash

dts="HV AD DLB NPH"
for _ in `seq 1000`; do
    echo "python train/MNet_1000_subj.py -dts $dts --does_channel_masking_exp &&"
    python train/MNet_1000_subj.py -dts $dts --does_channel_masking_exp 2>&1 | tee -a $0.log
done
