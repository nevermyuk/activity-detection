#!/usr/bin/env bash

python train.py \
-dataset TSU \
-split_setting "../data/TSU/smarthome_CS_51.json" \
-model PDAN \
-num_channel 512 \
-lr 0.0002 \
-kernelsize 3 \
-APtype map \
-epoch 139 \
-batch_size 1 \
-comp_info TSU_CS_RGB_PDAN \
-load_model 'False' \
-root '../data/TSU/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD' \
-trained_model_name 'the_best_model_ever'


