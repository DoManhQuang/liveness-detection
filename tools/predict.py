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

from core.utils import load_data, save_results_to_csv
from core.model import model_classification, model_mobile_v2_fine_tune

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_result", default="../runs", help="path save data")
parser.add_argument("--path_data", default="../public-test.data", help="path data image")
parser.add_argument("--name", default="data-name", help="data name save")
parser.add_argument("--model_path", default="../model.h5", help="model path")
parser.add_argument("--best_ckpt_path", default="../best-ckpt.h5", help="model check point path")
parser.add_argument("-v", "--version", default="0.1", help="version running")
parser.add_argument("--mode_weight", default="check-point", help="check-point or model-save")
parser.add_argument("--mode_model", default="name-model", help="mobi-v2")
parser.add_argument("--custom_objects", default=False, help="True or False")

args = vars(parser.parse_args())
path_data = args["path_data"]
version = args["version"]
model_path = args["model_path"]
best_ckpt_path = args["best_ckpt_path"]
name = args["name"]
save_result_path = args["save_result"]
mode_model = args["mode_model"]
mode_weight = args["mode_weight"]
custom_objects = args["custom_objects"]

print("==== START =======")

save_result_path = os.path.join(save_result_path, "results")
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

save_result_path = os.path.join(save_result_path, version)
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

print("loading data test ...")
dict_data_test_pub, labels_test_pub = load_data(path_data)
print("loading data test done")

print("loading model ...")
metrics = [
    # tfa.metrics.F1Score(num_classes=num_classes, average='weighted')
    'accuracy'

]

print("loading model ...")
model = None
if mode_model == "mobi-v2":
    model = model_mobile_v2_fine_tune(input_shape=dict_data_test_pub['shape'], num_class=1, activation='sigmoid')

if mode_weight == 'check-point':
    model.load_weights(best_ckpt_path)
elif mode_weight == 'model-save':
    if custom_objects:
        model = load_model(model_path, custom_objects={"F1Score": tfa.metrics.F1Score(num_classes=1, average="micro", threshold=0.5)})
    else:
        model = load_model(model_path)
print("loading model done")
model.summary()

labels_ids = []
res_min = []
res_max = []
res_mean = []
res_median = []
res_avg = []

print("model predict ...")
for i in tqdm(range(0, len(labels_test_pub))):
    y_predict = model.predict(np.array(dict_data_test_pub[labels_test_pub[i]]), verbose=0)
    labels_ids.append(labels_test_pub[i] + ".mp4")
    res_min.append(np.min(y_predict))
    res_max.append(np.max(y_predict))
    res_mean.append(np.mean(y_predict))
    res_median.append(np.median(y_predict))
    res_avg.append(np.average(y_predict))
    pass
print("model predict done")

print("folder save result : ", save_result_path)
save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_min
}, version=version, name=name + "-predict-min", directory=save_result_path)
print("Save results min done!!")

save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_max
}, version=version, name=name + "-predict-max", directory=save_result_path)
print("Save results max done!!")

save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_mean
}, version=version, name=name + "-predict-mean", directory=save_result_path)
print("Save results mean done!!")

save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_median
}, version=version, name=name + "-predict-median", directory=save_result_path)
print("Save results median done!!")

save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_avg
}, version=version, name=name + "-predict-avg", directory=save_result_path)
print("Save results avg done!!")


print("==== END =======")
