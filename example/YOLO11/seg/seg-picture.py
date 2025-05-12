from walnutpi_npu import YOLO11
import dataset_coco
import cv2
import numpy as np


model_path = "model/yolo11n-seg.nb"
picture_path = "image/000000371552.jpg"
output_path = ".result.jpg"


# 检测图片
yolo = YOLO11.YOLO11_SEG(model_path)
boxes = yolo.run(picture_path, 0.5, 0.5)

# 到图上画框
img = cv2.imread(picture_path)
for box in boxes:
    left_x = int(box.x - box.w / 2)
    left_y = int(box.y - box.h / 2)
    right_x = int(box.x + box.w / 2)
    right_y = int(box.y + box.h / 2)
    label = str(dataset_coco.label_names[box.label]) + " " + str(box.reliability)
    (label_width, label_height), bottom = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    cv2.rectangle(
        img,
        (left_x, left_y),
        (right_x, right_y),
        (255, 255, 0),
        2,
    )
    cv2.rectangle(
        img,
        (left_x, left_y - label_height * 2),
        (left_x + label_width, left_y),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        img,
        label,
        (left_x, left_y - label_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    mask_img = np.zeros_like(img)  # 生成一张与原图大小相同的纯黑图片
    mask_img[box.mask > 200] = (0, 255, 0)  # 将mask颜色值大于200的像素都改为绿色
    img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)  # 将mask_img与原图叠加

cv2.imwrite(output_path, img)
