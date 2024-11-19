#!/bin/bash

PREFIX='deepak8'
MODEL_CONFIG='deepak8_pf.py'
DATA_CONFIG='pf_features.yaml'
PATH_TO_SAMPLES='/eos/user/c/coli/public/weaver-benchmark/top_tagging/samples'

python train.py --predict \
 --data-test ${PATH_TO_SAMPLES}/prep/top_test_*.root \
 --num-workers 3 \
 --data-config /eos/user/d/disidiro/weaver-benchmark/top_tagging/data/${DATA_CONFIG} \
 --network-config /eos/user/d/disidiro/weaver-benchmark/top_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX}_best_epoch_state.pt \
 --gpus 0 --batch-size 512 \
 --predict-output output/${PREFIX}_predict.root