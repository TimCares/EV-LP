#!/bin/bash

model=""
model_name=""

for cfg in retrieval_coco retrieval_flickr30k; do
    python train.py --config-path ../configs/fine_tuning --config-name $cfg \
        pretrained.model_path=$model pretrained.model_name=$model_name
done
