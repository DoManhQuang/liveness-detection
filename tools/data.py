import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.model_selection import train_test_split
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.utils import load_data_image_directory, save_dump

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--save", default="../dataset/train", help="path save data")
parser.add_argument("--path_labels", default="../dataset/train/label.csv", help="path labels csv")
parser.add_argument("--path_data", default="../dataset/train/frame-video", help="path data image")
parser.add_argument("--height", default=300, type=int, help="image height")
parser.add_argument("--width", default=100, type=int, help="image width")
parser.add_argument("--img_count", default=10, type=int, help="image count")
parser.add_argument("--name", default="data-name", help="data name save")
parser.add_argument("--mode_color", default="gray", help="color image [gray | rgb]")

args = vars(parser.parse_args())
path_save = args["save"]
path_label = args["path_labels"]
path_data = args["path_data"]
img_height = args["height"]
img_width = args["width"]
cnt_image = args["img_count"]
data_name = args["name"]
mode_color = args["mode_color"]

print("======START=========")
labels_csv = np.array(pd.read_csv(path_label, usecols=["fname", "liveness_score"]))
labels_csv_ids = labels_csv[:, 0]
labels_csv_score = labels_csv[:, 1]
print("labels_csv_ids : ", labels_csv_ids)
print("labels_csv_score : ", labels_csv_score)

x_train, x_test, y_train, y_test = train_test_split(labels_csv_ids, labels_csv_score, test_size=0.2,
                                                    random_state=1000, shuffle=True, stratify=labels_csv_score)

data_train = np.concatenate(([x_train], [y_train]), axis=0).T
data_test = np.concatenate(([x_test], [x_test]), axis=0).T
print("TRAIN : ", data_train.shape)
print("TEST : ", data_test.shape)

print("=====Processing data train ========")
train_data, labels_train = load_data_image_directory(labels_csv=data_train, path_folder_data=path_data, img_height=img_height,
                                                     img_width=img_width, cnt_image=cnt_image, mode=mode_color)
print("=====Processing data train Done !!========")
print("=====Processing data test ========")
test_data, labels_test = load_data_image_directory(labels_csv=data_train, path_folder_data=path_data, img_height=img_height,
                                                   img_width=img_width, cnt_image=cnt_image, mode=mode_color)
print("=====Processing data test Done !!========")

print("save data ...")
save_dump(os.path.join(path_save, data_name + "-train.data"), data=train_data, labels=labels_train)
save_dump(os.path.join(path_save, data_name + "-test.data"), data=test_data, labels=labels_test)
print("save data done!!")
print("folder data train : ", os.path.join(path_save, data_name + "-train.data"))
print("folder data test : ", os.path.join(path_save, data_name + "-test.data"))
print("=========END!!===========")
