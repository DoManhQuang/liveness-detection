##  liveness-detection
Zalo challenge topic liveness-detection

BTC vui lòng cấu hình thư mục data chứa bộ private_test giống như bộ public_test và public_test_2
    
    zalo
    ├───data
    │   └───private_test
    │       └───videos
    └───results

BTC vui lòng không sử dụng --gpus "device=0" ở các lệnh docker, vì mình run thì thấy bị lỗi. BTC có thể run nếu không lỗi.
```shell
# Error use --gpus "device=0"
Node: 'model/conv2d/Relu'
DNN library is not found.
         [[{{node model/conv2d/Relu}}]] [Op:__inference_predict_function_772]
```

BTC follows commandline:
```shell
docker run --network host \
-it --name zac2022 nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 /bin/bash
```
```shell
docker cp /home/quangdm/zalo/zac2022/liveness-detection/. zac2022:/code/
```

```shell
docker commit -c zac2022 zac2022:v1
```

```shell
docker run \
-v /home/quangdm/zalo/zac2022/data/:/code/data \
-v /home/quangdm/zalo/zac2022/results/:/code/result \
zac2022:v1 \
/bin/bash /code/predict.sh \
--name zac2022
```

```shell
docker run -it -p9777:9777 \
-v /home/quangdm/zalo/zac2022/data/:/code/data \
-v /home/quangdm/zalo/zac2022/results/:/code/result \
zac2022:v1 /bin/bash /code/start_jupyter.sh
```
Xin cám ơn!