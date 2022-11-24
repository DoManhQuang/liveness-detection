import os
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.utils import face_detect

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--path_root", default="../dataset/train", help="path data")
parser.add_argument("--proto_path", default="../face-mask-detection/face_detector/deploy.prototxt", help="path proto")
parser.add_argument("--weights_path", default="../face_detector/res10_300x300_ssd_iter_140000.caffemodel", help="path weights")
parser.add_argument("--folder_face", default="face-video", help="folder face video output")
parser.add_argument("--folder_frame", default="frame-video", help="folder frame video output")
args = vars(parser.parse_args())
proto_path = args["proto_path"]
weights_path = args["weights_path"]
folder_face = args["folder_face"]
folder_frame = args["folder_frame"]
root = args["path_root"]

print("=======START============")
print("loading face net....")
face_net = cv2.dnn.readNet(proto_path, weights_path)
print("loading face net done!!")

folder_face = os.path.join(root, folder_face)
if not os.path.exists(folder_face):
    os.mkdir(folder_face)
    print("\ncreated folder : ", folder_face)

print("get list folder frame video")
folder_frame_video = os.path.join(root, folder_frame)
lst_folder_ids = os.listdir(folder_frame_video)
print("get list folder frame video done...")

zero_face = []
print("======== start ex-face ===========")
for i in range(0, len(lst_folder_ids)):

    print(os.path.join(folder_face, lst_folder_ids[i]))
    if not os.path.exists(os.path.join(folder_face, lst_folder_ids[i])):
        os.mkdir(os.path.join(os.path.join(folder_face, lst_folder_ids[i])))
        print("\ncreated folder : ", os.path.join(os.path.join(folder_face, lst_folder_ids[i])))

    lst_file_frame = os.listdir(os.path.join(folder_frame_video, lst_folder_ids[i]))

    print("lst_file_frame : ", lst_file_frame)

    for file_frame in lst_file_frame:
        frame_img = cv2.imread(os.path.join(folder_frame_video, lst_folder_ids[i], file_frame))
        file_frame_ids = file_frame.split(".")[0]
        faces = face_detect(frame=frame_img, face_net=face_net, dsize=(100, 100))

        if len(faces) > 0:
            print("save faces ...", np.array(faces).shape)
            for cnt_face in range(0, len(faces)):
                face_img_file = file_frame_ids + "-f-" + str(cnt_face) + ".jpg"
                cv2.imwrite(os.path.join(os.path.join(folder_face, lst_folder_ids[i]), face_img_file), faces[i])
            print("save faces done!!")
    if len(os.listdir(os.path.join(folder_face, lst_folder_ids[i]))) == 0:
        zero_face.append(lst_folder_ids[i])
    else:
        print("DONE")
print("Zero face list folder : ", zero_face)
print("======== start ex-face done===========")
print("=======END============")