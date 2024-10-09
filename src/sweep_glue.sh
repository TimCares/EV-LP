#!/bin/bash

model=""
model_name=""

for cfg in cola mnli mrpc qnli qqp rte sst stsb wnli; do
    python run_unimodal_train.py --config-path ../configs/fine_tuning --config-name $cfg model.model_path=$model model.model_name=$model_name
done
