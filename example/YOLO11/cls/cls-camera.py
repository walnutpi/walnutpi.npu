from walnutpi_npu import YOLO11
from walnutpi_npu import YOLO11
import dataset_ImageNet
import cv2
import os

os.environ["DISPLAY"] = ":0.0"

model_path = "model/yolo11n-cls.nb"
yolo = YOLO11.YOLO11_CLS(model_path)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 设置为1080p
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置长度

while True:
    # 读取一帧图像并显示出来
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if not yolo.is_running:
        yolo.run_async(img)
    result = yolo.get_result()
    index = 0
    if result is not None:
        for i in result.top5:
            show_string = "{:f} {:s}".format(
                i.reliability,
                dataset_ImageNet.label_names[i.label],
            )
            index += 1

            cv2.putText(
                img,
                show_string,
                (10, 30 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 200),
                2,
            )

    cv2.imshow("result", img)
    cv2.waitKey(1)
