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

args = vars(parser.parse_args())
path_save = args["save"]
path_label = args["path_labels"]
path_data = args["path_data"]

data_train, data_test, labels_train, labels_test = load_data_image_directory(path_folder_data=path_data, path_labels=path_label,
                                                                             img_height=300, img_width=100, cnt_image=5)

print("TRAIN : ", data_train.shape, "-", labels_train.shape)
print("TEST : ", data_test.shape, "-", labels_test.shape)

save_dump(os.path.join(path_save, "data-train.data"), data=data_train, labels=labels_train)
save_dump(os.path.join(path_save, "data-test.data"), data=data_test, labels=labels_test)
