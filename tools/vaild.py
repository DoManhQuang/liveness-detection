import os
import sys
import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from core.custom_metrics import equal_error_rate
from core.utils import load_data, write_score
from core.model import model_classification, model_mobile_v2_fine_tune

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-rp", "--result_path", default="../runs/results", help="path result ")
parser.add_argument("--path_data", default="../public-test.data", help="path data image")
parser.add_argument("--name", default="data-name", help="data name save")
parser.add_argument("--model_path", default="../model.h5", help="model path")
parser.add_argument("--best_ckpt_path", default="../best-ckpt.h5", help="model check point path")
parser.add_argument("-v", "--version", default="0.1", help="version running")
parser.add_argument("--mode_weight", default="check-point", help="check-point or model-save")
parser.add_argument("--mode_model", default="name-model", help="mobi-v2")
parser.add_argument("--custom_objects", default=False, help="True or False")
parser.add_argument("-test", "--test_data_path", default="../dataset/train/data-300x100-5-v1-test.data", help="data test")

args = vars(parser.parse_args())
path_data = args["path_data"]
version = args["version"]
model_path = args["model_path"]
best_ckpt_path = args["best_ckpt_path"]
name = args["name"]
result_path = args["result_path"]
mode_model = args["mode_model"]
mode_weight = args["mode_weight"]
custom_objects = args["custom_objects"]
model_name = args["name_model"]
test_path = args["test_data_path"]

print("==== START =======")
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)


result_path = os.path.join(result_path, model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)

print("loadin data test .....")
global_dataset_test, global_labels_test = load_data(test_path)
ip_shape = global_dataset_test[0].shape
print("loadin data test done!!")

print("loading model ...")
model = None
if mode_model == "mobi-v2":
    model = model_mobile_v2_fine_tune(input_shape=ip_shape, num_class=1, activation='sigmoid')

if mode_weight == 'check-point':
    print("loading weight model ...")
    if custom_objects:
        model = load_model(model_path, custom_objects={"F1Score": tfa.metrics.F1Score(num_classes=1, average="micro", threshold=0.5)})
    else:
        model = load_model(model_path)
    model.load_weights(best_ckpt_path)
    print("loading weight model done!!")
elif mode_weight == 'model-save':
    if custom_objects:
        model = load_model(model_path, custom_objects={"F1Score": tfa.metrics.F1Score(num_classes=1, average="micro", threshold=0.5)})
    else:
        model = load_model(model_path)
print("loading model done")
model.summary()


print("testing model.....")
y_predict = model.predict(global_dataset_test)
y_target = []

for score in y_predict:
    if score >= 0.5:
        y_target.append(1)
    else:
        y_target.append(0)

print("save results ......")
file_result = model_name + version + "score.txt"

write_score(path=os.path.join(result_path, file_result),
            mode_write="a",
            rows="STT",
            cols=['AUC', 'F1', 'Acc', 'EER'])

write_score(path=os.path.join(result_path, file_result),
            mode_write="a",
            rows="results",
            cols=np.around([roc_auc_score(global_labels_test, y_predict, average='micro'),
                            f1_score(global_labels_test, y_target, average='micro'),
                            accuracy_score(global_labels_test, y_target),
                            equal_error_rate(y_true=global_labels_test, y_predict=y_target)], decimals=4))
print("save results done!!")
