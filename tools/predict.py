import os
import shutil
import sys
import numpy as np
from tqdm import tqdm
from keras.models import load_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) == "/":
    ROOT = "/code"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.utils import load_data, save_results_to_csv
print("ROOT : ", ROOT)
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
parser.add_argument("--save_submit", default="../results", help="path save data")

args = vars(parser.parse_args())
save_submit = args["save_submit"]
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
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

save_result_path = os.path.join(save_result_path, "results")
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

save_result_path = os.path.join(save_result_path, version)
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

if not os.path.exists(save_submit):
    os.mkdir(save_submit)
    print("created folder :", save_submit)

print("loading data test ...")
dict_data_test_pub, labels_test_pub = load_data(path_data)
print("loading data test done")

print("loading model ...")
model = None
if mode_weight == 'check-point':
    print("loading weight model ...")
    if custom_objects:
        model = load_model(model_path, compile=False)
    else:
        model = load_model(model_path, compile=False)
    model.load_weights(best_ckpt_path)
    print("loading weight model done!!")
elif mode_weight == 'model-save':
    if custom_objects:
        model = load_model(model_path, compile=False)
    else:
        model = load_model(model_path, compile=False)
print("loading model done")
model.summary()

labels_ids = []
res_median = []
# res_min = []
# res_max = []
# res_mean = []
# res_avg = []

print("model predict ...")
for i in tqdm(range(0, len(labels_test_pub))):
    y_predict = model.predict(np.array(dict_data_test_pub[labels_test_pub[i]]), verbose=0)
    labels_ids.append(labels_test_pub[i] + ".mp4")
    res_median.append(np.median(y_predict))
    # res_min.append(np.min(y_predict))
    # res_max.append(np.max(y_predict))
    # res_mean.append(np.mean(y_predict))
    # res_avg.append(np.average(y_predict))
    pass

# y_target = []
# for score in res_median:
#     if score > 0.5:
#         y_target.append(1)
#     else:
#         y_target.append(0)
print("model predict done")
print("folder save result : ", save_result_path)
save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_median
}, version=version, name=name + "-predict-median", directory=save_result_path)

print("folder save result submit : ", save_submit)
save_results_to_csv(dict_results={
    "fname": labels_ids,
    "liveness_score": res_median
}, version=version, name=name + "-predict-median-submit", directory=save_submit)

print("Save results median done!!")
print("=====END====")
# save_results_to_csv(dict_results={
#     "fname": labels_ids,
#     "liveness_score": res_min
# }, version=version, name=name + "-predict-min", directory=save_result_path)
# print("Save results min done!!")
#
# save_results_to_csv(dict_results={
#     "fname": labels_ids,
#     "liveness_score": res_max
# }, version=version, name=name + "-predict-max", directory=save_result_path)
# print("Save results max done!!")
# save_results_to_csv(dict_results={
#     "fname": labels_ids,
#     "liveness_score": res_mean
# }, version=version, name=name + "-predict-mean", directory=save_result_path)
# print("Save results mean done!!")
# save_results_to_csv(dict_results={
#     "fname": labels_ids,
#     "liveness_score": res_avg
# }, version=version, name=name + "-predict-avg", directory=save_result_path)
# print("Save results avg done!!")

# save_results_to_csv(dict_results={
#     "fname": labels_ids,
#     "liveness_score": y_target
# }, version=version, name=name + "-predict-binary", directory=save_result_path)
# print("Save results binary done!!")
# print("==== END =======")
