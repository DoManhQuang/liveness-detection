#!/bin/bash

python /code/tools/ex-frame.py \
--path_root "/code/data/private_test" \
--folder_ex_frame "frame-video-private-test"

python /code/tools/dict-data-test.py \
--save "/code/data/private_test" \
--path_data "/code/data/private_test/frame-video-private-test" \
--height 300 --width 100 --img_count 5 \
--name "private-test-data-5x300x100x3" \
--mode_color "rgb"

python /code/tools/predict.py \
--name "model-g" \
--model_path "/code/runs/training/model-model-300x100x3-5-v3-v-0.3.h5" \
--path_data "/code/data/private_test/private-test-data-5x300x100x3.data" \
-v "v1.0" \
--save_result "/code/runs/submit" \
--mode_weight "model-save" \
--save_submit "/code/result"