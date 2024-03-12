#!/usr/bin/env bash

python inference.py \
-dataset TSU \
-model PDAN \
-APtype map \
-batch_size 1 \
-load_model 'PDAN_TSU_RGB' \
-root '../data/TSU/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD' \
-video_name "P14T02C04" \
> ./results/test_result.txt

