# HaUI Liveness detection [paper](https://jst-haui.vn/media/31/uffile-upload-no-title31211.pdf)
## Zalo challenge topic liveness-detection

### Problem statement:
In verification services related to face recognition (such as eKYC and face access control), the key question is whether the input face video is real (from a live person present at the point of capture), or fake (from a spoof artifact or lifeless body). Liveness detection is the AI problem to answer that question. In this challenge, participants will build a liveness detection model to classify if a given facial video is real or spoofed.
- Input: a video of selfie/portrait face with a length of 1-5 seconds (you can use any frames you like).
- Output: Liveness score in [0...1] (0 = Fake, 1 = Real).

### Training data:
1168 videos of faces with facemask, in which 598 are real and 570 are fake. Label file (Label.CSV) in which each row provides a video name and its label (0 = Fake, 1 = Real).

### Testing data:
- Public test: 350 videos of faces with facemask, without label file.
- Public test 2: 486 videos of faces with facemask, without label file.
- Private test: 1253 videos of faces with facemask, without label file.

### Results Final Leaderboard
private test score: 0.18382 - rank: 26

![image](https://user-images.githubusercontent.com/45645553/210210542-cf25e258-b208-48ba-854f-ab553d1a3ff0.png)

## Quick Start
### Install Lib
```shell
git clone https://github.com/DoManhQuang/liveness-detection.git
cd liveness-detection
pip install -r requirement.txt
```

### Extract frame from video
```shell
cd liveness-detection
python tools/ex-frame.py \
--path_root "path-video-format-zalo" \
--folder_ex_frame "name-of-folder"
```

### Processing data images (training)
```shell
cd liveness-detection
python tools/data.py \
--save "folder-save" \
--path_labels "path-label.csv" \
--path_data "path-name-of-folder" \
--height 300 --width 100 --img_count 5 \
--name "name-of-data" \
--mode_color "rgb" \
--test_size 0.1
```

### Processing data images (submit)
```shell
cd liveness-detection
python tools/dict-data-test.py \
--save "folder-save" \
--path_data "path-name-of-folder" \
--height 300 --width 100 --img_count 5 \
--name "name-of-data" \
--mode_color "rgb"
```

### Training model
```shell
cd liveness-detection
python tools/train.py \
-v "version-turn-training" \
-train "path-data-train.data" \
-test "path-data-test.data" \
-name "name-of-model-save" -ep 100 -bsize 16 -verbose 1 --mode_model "model-g" \
-resume "begin"
```

### Prediction (submit test)
```shell
%cd liveness-detection
python tools/predict.py \
--name "model-g" \
--model_path "path-model-save.h5" \
--path_data "path-data-test.data" \
-v "version-predict" \
--save_result "path-save-results" \
--mode_model "model-g" --mode_weight "model-save"
```

## References (Evaluate models performance)
### Running K-fold
```shell
cd liveness-detection
python tools/kfold.py \
-train "path-data-train.data" \
-test "path-data-test.data" \
-name "k-fold-name-of-model-save" \
--k_fold_path "path-save-runs-k-fold" \
--result_path "path-save-runs-results" \
--mode_model "model-g" \
-ep 100 -bsize 16 -verbose 1 \
-v "version-running-k-fold" \
-ck 1 -nk 10
```

### Valid data with models
```shell
cd liveness-detection
python tools/vaild.py \
--name "model-g" \
--model_path "path-model-save.h5" \
--path_data "path-data-test.data" \
-v "version-running-valid" \
--save_result "path-save-results" \
--mode_model "model-g" --mode_weight "model-save" --custom_objects True
```
