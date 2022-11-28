#!/bin/bash

python tools/ex-frame.py \
--path_root "./dataset/public_test_2" \
--folder_ex_frame "frame-video-private-test"

python tools/dict-data-test.py \
--save "./dataset/public_test_2" \
--path_data "./dataset/public_test_2/frame-video-private-test" \
--height 300 --width 100 --img_count 5 \
--name "private-test-data-5x300x100x3" \
--mode_color "rgb"

python tools/predict.py \
--name "model-g" \
--model_path "./runs/training/model-model-300x100x3-5-v3-v-0.3.h5" \
--path_data "./dataset/public_test_2/private-test-data-5x300x100x3.data" \
-v "v1.0" \
--save_result "./runs/submit" \
--mode_weight "model-save"