import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from keras.utils import img_to_array, load_img


def save_dump(file_path, data, labels):
    file = open(file_path, 'wb')
    # dump information to that file
    pickle.dump((data, labels), file)
    # close the file
    file.close()
    pass


def load_data(path_file):
    file = open(path_file, 'rb')
    # dump information to that file
    (pixels, labels) = pickle.load(file)
    # close the file
    file.close()
    # print(pixels.shape)
    # print(labels.shape)
    return pixels, labels


def load_data_image_directory(path_folder_data="../dataset", path_labels="labels.csv", img_height=600, img_width=300, cnt_image=5, mode='gray'):
    print("====Start Loading ... ====")
    data = []
    labels = []

    labels_csv = np.array(pd.read_csv(path_labels, usecols=["fname", "liveness_score"]))
    for i in tqdm(range(0, len(labels_csv))):
        folder_name = labels_csv[i][0].split(".")[0]
        folder_frame_path = os.path.join(path_folder_data, folder_name)
        lst_frame = os.listdir(folder_frame_path)
        for cnt in range(0, cnt_image):
            image = load_img(os.path.join(folder_frame_path, lst_frame[cnt]), target_size=(img_height, img_width))  # (img_height, img_width)
            image = img_to_array(image)
            if mode == 'gray':
                image = tf.image.rgb_to_grayscale(image).numpy()
            data.append(image)
            labels.append(labels_csv[i][1])

    data = np.array(data)
    labels = np.array(labels)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, shuffle=True,
                                                                        stratify=labels, random_state=10000)

    print("====End Done !!! ====")
    return data_train, data_test, labels_train, labels_test


def load_data_image_test(path_folder_data="../dataset", img_height=600, img_width=300, cnt_image=10, mode='gray'):
    print("====Start Loading ... ====")
    dict_ids = {}
    labels_ids = []
    lst_frame_video = os.listdir(path_folder_data)
    for i in tqdm(range(0, len(lst_frame_video))):
        folder_ids = lst_frame_video[i]
        folder_ids_path = os.path.join(path_folder_data, folder_ids)
        lst_frame = os.listdir(folder_ids_path)
        data_ids = []
        for cnt in range(0, cnt_image):
            image = load_img(os.path.join(folder_ids_path, lst_frame[cnt]), target_size=(img_height, img_width))  # (img_height, img_width)
            image = img_to_array(image)
            if mode == 'gray':
                image = tf.image.rgb_to_grayscale(image).numpy()
            data_ids.append(image)
            dict_ids['shape'] = np.array(image).shape
        dict_ids[folder_ids] = data_ids
        labels_ids.append(folder_ids)

    print("====End Done !!! ====")
    return dict_ids, np.array(labels_ids)


def get_callbacks_list(diractory,
                       status_tensorboard=True,
                       status_checkpoint=True,
                       status_earlystop=True,
                       file_ckpt="ghtk-spamham-weights-best-training-file.hdf5",
                       ckpt_monitor='val_accuracy',
                       ckpt_mode='max',
                       early_stop_monitor="val_accuracy",
                       early_stop_mode="max",
                       early_stop_patience=5):
    callbacks_list = []
    save_path = []
    if status_earlystop:
        # Early Stopping
        callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=early_stop_patience,
                                                               restore_best_weights=True, mode=early_stop_mode, verbose=1)
        callbacks_list.append(callback_early_stop)

    # create checkpoint
    if status_checkpoint:
        checkpoint_path = os.path.join(diractory, "checkpt")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        # file="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        file_path = os.path.join(checkpoint_path, file_ckpt)
        save_path.append(file_path)

        checkpoint_callback = ModelCheckpoint(file_path, monitor=ckpt_monitor, verbose=1,
                                              save_best_only=True, mode=ckpt_mode, save_weights_only=True)
        callbacks_list.append(checkpoint_callback)

    # Tensorflow Board
    if status_tensorboard:
        tensorboard_path = os.path.join(diractory, "tensorboard-logs")
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
        callbacks_list.append(tensorboard_callback)
        save_path.append(tensorboard_path)

    return callbacks_list, save_path


def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1), len(model_history.history[acc]) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    # plt.savefig('roc.png')


def plot_model_legend(model_history):
    # view
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation')
    plt.xlim(0, len(accuracy))
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()


def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cmx, annot=True, fmt='g')
    print(cmx_data)
    plt.show()


def save_results_to_csv(dict_results, directory="./results", name="predict", version="version-0.0"):
    df_res = pd.DataFrame(data=dict_results)
    file_name = name + "-" + version + "-result.csv"
    df_res.to_csv(os.path.join(directory, file_name), encoding="utf-8", index=False)
    print("SAVE DONE")
    pass


def write_score(path="test.txt", mode_write="a", rows="STT", cols=[1.0, 2.0, 3.0]):
    file = open(path, mode_write)
    file.write(str(rows) + "*")
    for col in cols:
        file.write(str(col) + "*")
    file.write("\n")
    file.close()
    pass


def set_gpu_limit(set_memory=5):
    memory_limit = set_memory * 1024
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)