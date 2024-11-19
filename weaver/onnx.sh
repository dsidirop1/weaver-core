#!/bin/bash

python train.py -c /eos/user/d/disidiro/weaver-benchmark/top_tagging/data/pf_points_features.yaml -n /eos/user/d/disidiro/weaver-benchmark/top_tagging/networks/particlenet_pf.py -m /eos/user/d/disidiro/weaver-core/weaver/output/particlenet_best_epoch_state.pt --export-onnx particlenet.onnx