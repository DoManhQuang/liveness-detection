import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from keras.preprocessing.image import ImageDataGenerator
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf
import tensorflow_addons as tfa
from core.utils import load_data, get_callbacks_list, set_gpu_limit
from core.model import model_classification, model_mobile_v2_fine_tune
from core.fas_base_01 import created_model_fas_01
from core.custom_metrics import equal_error_rate


# # Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--memory", default=0, type=int, help="set gpu memory limit")
parser.add_argument("-v", "--version", default="version-0.2", help="version running")
parser.add_argument("-rp", "--result_path", default="../runs/results", help="path result ")
parser.add_argument("-tp", "--training_path", default="../runs/training", help="path training model")
parser.add_argument("-ep", "--epochs", default=1, type=int, help="epochs training")
parser.add_argument("-bsize", "--bath_size", default=8, type=int, help="bath size training")
parser.add_argument("-verbose", "--verbose", default=1, type=int, help="verbose training")
parser.add_argument("-train", "--train_data_path", default="../dataset/train/data-300x100-5-v1-train.data", help="data training")
parser.add_argument("-val", "--val_data_path", default="../dataset/train/data-valid.data", help="data val")
parser.add_argument("-test", "--test_data_path", default="../dataset/train/data-300x100-5-v1-test.data", help="data test")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
parser.add_argument("--mode_model", default="name-model", help="mobi-v2")
args = vars(parser.parse_args())

# Set up parameters
version = args["version"]
training_path = args["training_path"]
result_path = args["result_path"]
epochs = args["epochs"]
bath_size = args["bath_size"]
verbose = args["verbose"]
gpu_memory = args["memory"]
train_path = args["train_data_path"]
val_path = args["val_data_path"]
test_path = args["test_data_path"]
model_name = args["name_model"]
mode_model = args["mode_model"]

print("=========Start=========")
if gpu_memory > 0:
    set_gpu_limit(int(gpu_memory))  # set GPU

print("=====loading dataset ...======")
global_dataset_train, global_labels_train = load_data(train_path)
# global_dataset_val, global_labels_val = load_data(val_path)
global_dataset_test, global_labels_test = load_data(test_path)

print("TRAIN : ", global_dataset_train.shape, " - ", global_labels_train.shape)
# print("VAL : ", global_dataset_val.shape, " - ", global_labels_val.shape)
print("TEST : ", global_dataset_test.shape, " - ", global_labels_test.shape)

print("=======loading dataset done!!=======")
# num_classes = len(np.unique(global_labels_train))
ip_shape = global_dataset_train[0].shape
metrics = [
    equal_error_rate,
    # 'accuracy',
    # tfa.metrics.F1Score(num_classes=1, average="micro", threshold=0.5)
    'binary_accuracy'
]

dict_model = {
    "mobi-v2": model_mobile_v2_fine_tune(input_shape=ip_shape, num_class=1, activation='sigmoid'),
    "model-g": model_classification(input_layer=ip_shape, num_class=1, activation='sigmoid'),
    "fas-v1": created_model_fas_01(input_shape=ip_shape, number_class=1, activation='sigmoid')
}
model = dict_model[mode_model]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=metrics)
model.summary()
weights_init = model.get_weights()
print("model loading done!!")

# created folder

if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

training_path = os.path.join(training_path, model_name)
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

training_path = os.path.join(training_path, version)
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

if not os.path.exists(os.path.join(training_path, 'model-save')):
    os.makedirs(os.path.join(training_path, 'model-save'))
    print("created folder : ", os.path.join(training_path, 'model-save'))


if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)


result_path = os.path.join(result_path, model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)

# training

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

file_ckpt_model = "best-weights-training-file-" + model_name + "-" + version + ".h5"
# callback list
callbacks_list, save_list = get_callbacks_list(training_path,
                                               status_tensorboard=True,
                                               status_checkpoint=False,
                                               status_earlystop=True,
                                               file_ckpt=file_ckpt_model,
                                               ckpt_monitor='val_binary_accuracy',
                                               ckpt_mode='max',
                                               early_stop_monitor="val_equal_error_rate",
                                               early_stop_mode="min",
                                               early_stop_patience=10
                                               )
print("Callbacks List: ", callbacks_list)
print("Save List: ", save_list)

print("===========Training==============")

train_datagen.fit(global_dataset_train)
test_datagen.fit(global_dataset_test)

model.set_weights(weights_init)
# model_history = model.fit(global_dataset_train, global_labels_train, epochs=epochs, batch_size=bath_size,
#                           verbose=verbose, validation_data=(global_dataset_test, global_labels_test),
#                           shuffle=True, callbacks=callbacks_list)

model.fit(train_datagen.flow(global_dataset_train, global_labels_train, batch_size=bath_size),
          validation_data=test_datagen.flow(global_dataset_test, global_labels_test, batch_size=bath_size),
          epochs=epochs, shuffle=True, callbacks=callbacks_list)

print("===========Training Done !!==============")
model_save_file = "model-" + model_name + "-" + version + ".h5"
model.save(os.path.join(training_path, 'model-save', model_save_file), save_format='h5')
print("Save model done!!")

# evaluate model
scores = model.evaluate(global_dataset_test, global_labels_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("History training loading ...")
cmd = 'tensorboard --logdir "path-tensorboard-logs/"'
print("CMD: ", cmd)
for file_log in save_list:
    print("file_log: ", file_log)
print("============END=============")
