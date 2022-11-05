import os
import cv2
from tqdm import tqdm
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

root = "../dataset/public"
folder_frame = "frame-video"
if not os.path.exists(os.path.join(root, folder_frame)):
    os.mkdir(os.path.join(root, folder_frame))
    print("\ncreated folder : ", os.path.join(root, folder_frame))

lst_name_video = os.listdir(os.path.join(root, "videos"))
print(lst_name_video)

for i in range(0, len(lst_name_video)):
    cap = cv2.VideoCapture(os.path.join(root, "videos", lst_name_video[i]))
    folder_frame_ids = lst_name_video[i].split(".")[0]
    if not os.path.exists(os.path.join(root, folder_frame, folder_frame_ids)):
        os.mkdir(os.path.join(root, folder_frame, folder_frame_ids))
        print("\ncreated folder : ", os.path.join(root, folder_frame, folder_frame_ids))

    # Check if camera opened successfully
    if not cap.isOpened():
        print("\nError opening video file")

    for cnt in tqdm(range(0, 10)):
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Saving the image
                filename = os.path.join(root, folder_frame, folder_frame_ids, f"{folder_frame_ids}-{cnt}.jpg")
                # print(filename)
                cv2.imwrite(filename, frame)
