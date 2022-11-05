import os
import sys
import pandas as pd
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


lst_name_video = os.listdir("../dataset/public/videos")
print(lst_name_video)
label_video = [0 for i in range(0, len(lst_name_video))]
dict_res = {
    "fname": lst_name_video,
    "liveness_score": label_video
}
df = pd.DataFrame(dict_res)
df.to_csv("../results/Predict.csv", index=False)
