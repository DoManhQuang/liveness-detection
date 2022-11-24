import os
import cv2
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def save_image(video_cap, file_name, time_msec):
    video_cap.set(cv2.CAP_PROP_POS_MSEC, time_msec)
    ret, frame = video_cap.read()
    if ret:
        # print(filename)
        cv2.imwrite(file_name, frame)
    pass


def get_duration_in_seconds(video_cap):
    return video_cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_cap.get(cv2.CAP_PROP_FPS)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--path_root", default="../dataset/train", help="path data")
# parser.add_argument("--img_cnt", default=5, type=int, help="number frame in video")
# parser.add_argument("--seconds", default=5, type=int, help="seconds for video")
# parser.add_argument("--minutes", default=0, type=int, help="minutes for video")
parser.add_argument("--folder_ex_frame", default="frame-video", help="frame video")
args = vars(parser.parse_args())

root = args["path_root"]
# img_cnt = args["img_cnt"]
# seconds = args["seconds"]
# minutes = args["minutes"]
folder_frame = args["folder_ex_frame"]
error_ex = []
print("========START===========")
if not os.path.exists(os.path.join(root, folder_frame)):
    os.mkdir(os.path.join(root, folder_frame))
    print("\ncreated folder : ", os.path.join(root, folder_frame))

lst_name_video = os.listdir(os.path.join(root, "videos"))
print(lst_name_video)
for i in range(0, len(lst_name_video)):
    folder_frame_ids = lst_name_video[i].split(".")[0]
    if not os.path.exists(os.path.join(root, folder_frame, folder_frame_ids)):
        os.mkdir(os.path.join(root, folder_frame, folder_frame_ids))
        print("\ncreated folder : ", os.path.join(root, folder_frame, folder_frame_ids))

    video = cv2.VideoCapture(os.path.join(root, "videos", lst_name_video[i]))
    if not video.isOpened():
        print("\nError opening video file")

    minutes = 0
    seconds = get_duration_in_seconds(video_cap=video)
    param = [0.15, 0.25, 0.5, 0.75, 0.95]
    for par in tqdm(range(0, len(param))):
        if video.isOpened():
            t_msec = 1000 * (minutes * 60 + seconds * param[par])
            filename = os.path.join(root, folder_frame, folder_frame_ids, f"{folder_frame_ids}-{seconds * param[par]}.jpg")
            save_image(video_cap=video, file_name=filename, time_msec=t_msec)

    lst_file = os.listdir(os.path.join(root, folder_frame, folder_frame_ids))
    if len(lst_file) == len(param):
        print("DONE!!")
    else:
        error_ex.append(folder_frame_ids)
        print("ERROR Ex-Frame: ", os.path.join(root, folder_frame, folder_frame_ids))
    pass

print("ERROR Ex-Frame list: ", error_ex)
print("========END===========")
