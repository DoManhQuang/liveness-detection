import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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
parser.add_argument("--mode_color", default="gray", help="color image")


args = vars(parser.parse_args())
path_save = args["save"]
path_label = args["path_labels"]
path_data = args["path_data"]
img_height = args["height"]
img_width = args["width"]
cnt_image = args["img_count"]
data_name = args["name"]
mode_color = args["mode_color"]

data_train, data_test, labels_train, labels_test = load_data_image_directory(path_folder_data=path_data, path_labels=path_label,
                                                                             img_height=img_height, img_width=img_width, cnt_image=cnt_image, mode=mode_color)

print("TRAIN : ", data_train.shape, "-", labels_train.shape)
print("TEST : ", data_test.shape, "-", labels_test.shape)

save_dump(os.path.join(path_save, data_name + "-train.data"), data=data_train, labels=labels_train)
save_dump(os.path.join(path_save, data_name + "-test.data"), data=data_test, labels=labels_test)
