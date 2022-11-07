import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.utils import load_data, save_results_to_csv
from core.model import model_classification

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_result", default="../runs", help="path save data")
parser.add_argument("--path_data", default="../dataset/public/data-300x100-5-v1-public-test.data", help="path data image")
parser.add_argument("--name", default="data-name", help="data name save")
parser.add_argument("--model_path", default="model.h5", help="model path")
parser.add_argument("--best_ckpt_path", default="../runs/training/best-weights-training-file-model-300x100-5-version-0.2-100ep.h5", help="model check point path")
parser.add_argument("-v", "--version", default="0.1", help="version running")

args = vars(parser.parse_args())
path_data = args["path_data"]
version = args["version"]
model_path = args["model_path"]
best_ckpt_path = args["best_ckpt_path"]
name = args["name"]
save_result_path = args["save_result"]

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
model = model_classification((300, 100, 1), num_class=1, activation='sigmoid')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=metrics)
model.summary()
model.load_weights(best_ckpt_path)
print("loading model done")
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