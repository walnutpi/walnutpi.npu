from walnutpi import YOLO11
import cv2

label_names = ["person"]

model_path = "model/yolo11n-pose.nb"
picture_path = "image/kun.jpg"
output_path = ".result.jpg"

# 检测图片
yolo = YOLO11.YOLO11_POSE(model_path)
boxes = yolo.run(picture_path, 0.5, 0.5)

# 到图上画框
img = cv2.imread(picture_path)
for box in boxes:
    left_x = int(box.x - box.w / 2)
    left_y = int(box.y - box.h / 2)
    right_x = int(box.x + box.w / 2)
    right_y = int(box.y + box.h / 2)
    label = str(label_names[box.label]) + " " + str(box.reliability)
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
    # 0 鼻子 1 左眼 2 右眼 3 左耳 4 右耳 5 左肩 6 右肩
    # 7 左手肘 8 右手肘 9 左手腕 10 右手腕 11 左髋部 12 右髋部
    # 13 左膝盖 14 右膝盖 15 左脚踝 16 右脚踝
    for i in box.keypoints:  # 绘制所有可见度够高的关键点
        if i.visibility > 0.5:
            cv2.circle(img, i.xy, 5, (0, 0, 200), -1)
    # 左右手连接
    cv2.line(img, box.keypoints[7].xy, box.keypoints[5].xy, (255, 0, 0), 8)
    cv2.line(img, box.keypoints[7].xy, box.keypoints[9].xy, (255, 0, 0), 8)
    cv2.line(img, box.keypoints[8].xy, box.keypoints[6].xy, (255, 0, 0), 8)
    cv2.line(img, box.keypoints[8].xy, box.keypoints[10].xy, (255, 0, 0), 8)

    # 中间身体相连
    cv2.line(img, box.keypoints[5].xy, box.keypoints[6].xy, (100, 255, 100), 8)
    cv2.line(img, box.keypoints[11].xy, box.keypoints[5].xy, (100, 255, 100), 8)
    cv2.line(img, box.keypoints[12].xy, box.keypoints[6].xy, (100, 255, 100), 8)
    cv2.line(img, box.keypoints[12].xy, box.keypoints[11].xy, (100, 255, 100), 8)

    # 左右脚连接
    cv2.line(img, box.keypoints[13].xy, box.keypoints[11].xy, (255, 255, 0), 8)
    cv2.line(img, box.keypoints[13].xy, box.keypoints[15].xy, (255, 255, 0), 8)
    cv2.line(img, box.keypoints[14].xy, box.keypoints[12].xy, (255, 255, 0), 8)
    cv2.line(img, box.keypoints[14].xy, box.keypoints[16].xy, (255, 255, 0), 8)


cv2.imwrite(output_path, img)
