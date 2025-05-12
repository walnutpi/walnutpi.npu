from walnutpi_npu import YOLO11
import dataset_coco
import cv2

path_model = "model/yolo11n.nb"
path_image = "image/bus.jpg"
output_filename = ".result.jpg"

# 读取图片
img = cv2.imread(path_image)

# 检测图片
yolo = YOLO11.YOLO11_DET(path_model)
boxes = yolo.run(img, 0.5, 0.45)

# 输出检测结果
print(f"boxes: {boxes.__len__()}")
for box in boxes:
    print(
        "{:f} ({:4d},{:4d}) w{:4d} h{:4d} {:s}".format(
            box.reliability,
            box.x,
            box.y,
            box.w,
            box.h,
            dataset_coco.label_names[box.label],
        )
    )

# 到图上画框
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

# 保存图片
cv2.imwrite(output_filename, img)
print(f"save to {output_filename}")
