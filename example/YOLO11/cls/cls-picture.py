from walnutpi_npu import YOLO11
import dataset_ImageNet
import cv2

model_path = "model/yolo11n-cls.nb"
picture_path = "image/banana.jpg"
output_path = ".result.jpg"

img = cv2.imread(picture_path)

# 检测图片
yolo = YOLO11.YOLO11_CLS(model_path)
result = yolo.run(img)

# 输出与绘制到图片上
index = 0
for i in result.top5:
    show_string = "{:f} {:s}".format(
        i.reliability,
        dataset_ImageNet.label_names[i.label],
    )
    print(show_string)
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

# 保存图片
cv2.imwrite(output_path, img)
