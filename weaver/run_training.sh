#!/bin/bash

PREFIX='mlp'
MODEL_CONFIG='mlp_pf.py'
DATA_CONFIG='pf_features.yaml'
PATH_TO_SAMPLES='/eos/home-c/coli/public/weaver-benchmark/top_tagging/samples'

python train.py \
 --data-train ${PATH_TO_SAMPLES}/prep/top_train_*.root \
 --data-val ${PATH_TO_SAMPLES}/prep/top_val_*.root \
 --fetch-by-file --fetch-step 1 --num-workers 3 \
 --data-config /eos/user/d/disidiro/weaver-benchmark/top_tagging/data/${DATA_CONFIG} \
 --network-config /eos/user/d/disidiro/weaver-benchmark/top_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX} \
 --gpus 0 --batch-size 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 --log output/${PREFIX}.train.log
