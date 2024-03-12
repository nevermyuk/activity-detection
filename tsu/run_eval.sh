#!/usr/bin/env bash

python evaluation.py \
-dataset TSU \
-split_setting "../data/TSU/smarthome_CS_51.json" \
-model PDAN \
-APtype map \
-batch_size 1 \
-load_model 'PDAN_TSU_RGB' \
-root '../data/TSU/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD' 

