import os
import numpy as np
import cv2
from typing import List
from walnutpi_npu import NPU 
import time
import threading


class YOLO_RESULT_DET:
    x: int
    y: int
    w: int
    h: int
    xywh: np.ndarray
    label: int  # 类别索引
    reliability: float  # 置信度
    index_in_all_boxes: int


class YOLO_RESULT_OBB(YOLO_RESULT_DET):
    angle: float  # 旋转角度

    def _rotate_point(self, cx, cy, x, y, angle):
        """旋转点(x, y)围绕中心点(cx, cy)旋转angle弧度"""
        s, c = np.sin(angle), np.cos(angle)
        x_new = c * (x - cx) - s * (y - cy) + cx
        y_new = s * (x - cx) + c * (y - cy) + cy
        return int(x_new), int(y_new)

    def get_top_left(self):
        """获取旋转后的左上角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x - half_w, self.y - half_h, self.angle
        )

    def get_bottom_left(self):
        """获取旋转后的左下角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x - half_w, self.y + half_h, self.angle
        )

    def get_top_right(self):
        """获取旋转后的右上角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x + half_w, self.y - half_h, self.angle
        )

    def get_bottom_right(self):
        """获取旋转后的右下角坐标"""
        half_w, half_h = self.w / 2, self.h / 2
        return self._rotate_point(
            self.x, self.y, self.x + half_w, self.y + half_h, self.angle
        )


class YOLO_RESULT_SEG(YOLO_RESULT_DET):
    contours: list  # 边界点的坐标，形式是 ((x1, y1)，....)
    mask: cv2.typing.MatLike  # 一张单通道图片，被识别为物体的区域为255，背景为0
    _raw_mask: np.ndarray


class _YOLO_KEYPOINT:
    xy = (0, 0)
    visibility: float


class YOLO_RESULT_POSE(YOLO_RESULT_DET):
    keypoints: List[_YOLO_KEYPOINT] = []  # 各个关键点的坐标


class _YOLO_RESULT_CLS_INDEX:
    label: int  # 类别索引
    reliability: float  # 置信度

    def __init__(self, label=0, reliability=0):
        self.label = label
        self.reliability = reliability


class YOLO_RESULT_CLS:
    # TOP5包含了置信度排名前5的类别
    top5 = [
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
        _YOLO_RESULT_CLS_INDEX(),
    ]
    # all是一个数组，包含所有类别的置信度，all[35]代表类别35的置信度，以此类推
    all = np.zeros(1000)


def desigmoid(x):
    return -np.log(1.0 / x - 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class _YOLO_BASE:
    has_result = False
    is_running = False
    npu: NPU.awnn
    model_img_size: int
    nms_threshold = 0.45  # nms阈值
    results = []

    class _speed:
        ms_pre_process: int = 0  # 前处理耗时
        ms_post_process: int = 0  # 后处理耗时
        ms_inference: int = 0  # 推理耗时

    speed = _speed()

    def __init__(self, path: str):
        """
        初始化
        @path: 模型路径
        @model_img_size: 模型设定的图片输入尺寸
        """
        self.npu = NPU.awnn(path)

    def run(self, img, reliability_threshold=0.5, nms_threshold=0.5):
        """
        检测图片，阻塞直到检测完成，返回检测结果
        @path: 图片路径
        """
        self.run_async(img, reliability_threshold, nms_threshold)
        while not self.has_result:
            time.sleep(0.001)
        return self.results

    def run_async(self, img, reliability_threshold=0.5, nms_threshold=0.5):
        """
        检测图片，立即返回，不阻塞
        @path: 图片路径
        """
        if not self.is_running:
            self.nms_threshold = nms_threshold
            self.is_running = True

            thread = threading.Thread(
                target=self.thread_async_run,
                args=(
                    img,
                    reliability_threshold,
                ),
            )
            thread.start()

    def thread_async_run(self, img, reliability_threshold):
        time_point = int(time.time() * 1000)

        if not self.npu.is_async_running():
            data = self.pre_process(img)
            self.speed.ms_pre_process = int(time.time() * 1000) - time_point
            time_point = int(time.time() * 1000)
            self.npu.run_async(data)

        while self.npu.is_async_running():
            time.sleep(0.001)
            # print(f"{time.time()}", end="\r")
        # print(f"                                    ", end="\r")

        self.speed.ms_inference = int(time.time() * 1000) - time_point
        time_point = int(time.time() * 1000)

        self.results = self.post_process(reliability_threshold)
        self.speed.ms_post_process = int(time.time() * 1000) - time_point
        time_point = int(time.time() * 1000)

        self.has_result = True
        self.is_running = False

    def get_result(self):
        self.has_result = False
        return self.results

    def resize_img2model(self, img: cv2.typing.MatLike):
        """
        将图片从原始尺寸缩小到模型输入尺寸
        """
        ih, iw, _ = img.shape
        w, h = self.model_img_size, self.model_img_size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        new_img = np.full((h, w, 3), 128, dtype=np.uint8)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_img[dy : dy + nh, dx : dx + nw, :] = cv2.resize(
            img, (nw, nh), interpolation=cv2.INTER_CUBIC
        )
        return new_img

    def resize_model2img(self, img: cv2.typing.MatLike):
        """
        将输入的模型尺寸图片放大回原始尺寸
        """
        raw_h, raw_w, _ = self.img_raw.shape
        img_h, img_w, ishape = img.shape
        scale = max(raw_w / img_w, raw_h / img_h)
        sraw_w = int(raw_w / scale)
        sraw_h = int(raw_h / scale)
        diff_h = int((img_h - sraw_h) / 2)
        dirr_w = int((img_w - sraw_w) / 2)
        new_img = np.full((sraw_h, sraw_w, ishape), 0, dtype=np.uint8)

        new_img[...] = img[diff_h : sraw_h + diff_h, dirr_w : sraw_w + dirr_w, :]
        new_img = cv2.resize(new_img, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        return new_img

    def pre_process(self, img):
        """图像前处理"""
        # 判断picture的数据类型是str吗
        if isinstance(img, str):
            if not os.path.isfile(img):
                raise FileNotFoundError("文件不存在")
            return self.pre_process(cv2.imread(img))
        if len(img.shape) == 2 or img.shape[2] == 1:
            self.img_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            self.img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_resize = self.resize_img2model(self.img_raw)

        # 转换数据格式
        img_nchw = np.transpose(self.img_resize, (2, 0, 1))
        img_nchw_uint8 = img_nchw.astype(np.uint8)
        return bytearray(img_nchw_uint8.tobytes())

    def nms_sorted_bboxes(
        self, boxes: List["YOLO_RESULT_DET"], nms_threshold: float = 0.2
    ) -> List["YOLO_RESULT_DET"]:
        if not boxes:
            return []

        boxes.sort(key=lambda box: box.reliability, reverse=True)

        picked = []
        areas = [(box.w * box.h) for box in boxes]

        for i in range(len(boxes)):
            if areas[i] == 0:
                continue

            picked.append(boxes[i])

            for j in range(i + 1, len(boxes)):
                if areas[j] == 0:
                    continue

                x1 = max(boxes[i].x - boxes[i].w / 2, boxes[j].x - boxes[j].w / 2)
                y1 = max(boxes[i].y - boxes[i].h / 2, boxes[j].y - boxes[j].h / 2)
                x2 = min(boxes[i].x + boxes[i].w / 2, boxes[j].x + boxes[j].w / 2)
                y2 = min(boxes[i].y + boxes[i].h / 2, boxes[j].y + boxes[j].h / 2)

                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                inter_area = w * h

                union_area = areas[i] + areas[j] - inter_area

                iou = inter_area / union_area

                if iou > nms_threshold:
                    areas[j] = 0

        return picked


class YOLO11_CLS(_YOLO_BASE):
    results = YOLO_RESULT_CLS()  # 识别到的分类结果
    model_img_size = 224

    def get_result(self) -> YOLO_RESULT_CLS:
        return super().get_result()

    def run(self, img) -> YOLO_RESULT_CLS:
        return super().run(img)

    def post_process(self, reliability_threshold):
        tensor = self.npu.output_buffer.get(0)
        tensor = sigmoid(tensor)

        top_5_indices = np.argsort(tensor)[-5:][::-1]
        ret = YOLO_RESULT_CLS()
        ret.all = tensor
        for i in range(5):
            index = top_5_indices[i]
            ret.top5[i] = _YOLO_RESULT_CLS_INDEX(index, tensor[index])
        return ret


class YOLO11_DET(_YOLO_BASE):
    results: List[YOLO_RESULT_DET] = []
    _result_type = YOLO_RESULT_DET
    model_img_size = 640

    def get_result(self) -> List[YOLO_RESULT_DET]:
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> List[YOLO_RESULT_DET]:
        return super().run(img, reliability_threshold, nms_threshold)

    def get_boxes(
        self, tensor_out, tensor_classes, reliability_threshold, stride
    ) -> List[YOLO_RESULT_DET]:
        de_releability_threshold = desigmoid(reliability_threshold)
        mask = tensor_classes > de_releability_threshold
        indices = np.argwhere(mask)
        box_per_line = int(self.model_img_size / stride)
        index_in_out = 0
        if stride == 8:
            index_in_out = 0
        if stride == 16:
            index_in_out = int(self.model_img_size * self.model_img_size / 64)
        if stride == 32:
            index_in_out = int(
                self.model_img_size * self.model_img_size / 64
                + self.model_img_size * self.model_img_size / 256
            )
        ret = []
        classes_reliability = sigmoid(tensor_classes[mask])
        for i, cls in zip(indices, classes_reliability):
            postion = index_in_out + i[1] * box_per_line + i[2]
            re = self._result_type()
            re.index_in_all_boxes = postion
            re.x = int(tensor_out[0, 0, postion])
            re.y = int(tensor_out[0, 1, postion])
            re.w = int(tensor_out[0, 2, postion])
            re.h = int(tensor_out[0, 3, postion])
            re.xywh = tensor_out[0, 0:4, postion]
            re.reliability = cls
            re.label = i[3]
            ret.append(re)

        return ret

    def scale_boxes(self, boxes) -> List[YOLO_RESULT_DET]:
        # 缩放坐标到与原始图像一致
        original_height, original_width, _ = self.img_raw.shape
        scale = min(
            self.model_img_size / original_width, self.model_img_size / original_height
        )
        pad_x = (self.model_img_size - original_width * scale) / 2
        pad_y = (self.model_img_size - original_height * scale) / 2

        for box in boxes:
            box.x = int((box.x - pad_x) / scale)
            box.y = int((box.y - pad_y) / scale)
            box.w = int(box.w / scale)
            box.h = int(box.h / scale)

        return boxes

    def post_process(self, reliability_threshold):
        # self.npu.save_tensor(".")
        tensor_out = self.npu.output_buffer.get(0)
        tensor_8 = self.npu.output_buffer.get(1)
        tensor_16 = self.npu.output_buffer.get(2)
        tensor_32 = self.npu.output_buffer.get(3)

        box_per_line_8 = int(self.model_img_size / 8)
        box_per_line_16 = int(self.model_img_size / 16)
        box_per_line_32 = int(self.model_img_size / 32)
        box_count_8 = int(box_per_line_8**2)
        box_count_16 = int(box_per_line_16**2)
        box_count_32 = int(box_per_line_32**2)
        box_count_out = box_count_8 + box_count_16 + box_count_32

        tensor_shape1_out = int(tensor_out.__len__() / box_count_out)
        tensor_shape1_stride = int(tensor_8.__len__() / box_count_8)
        tensor_out = np.reshape(
            tensor_out,
            (1, tensor_shape1_out, box_count_out),
        )
        tensor_8 = np.reshape(
            tensor_8,
            (1, tensor_shape1_stride, box_per_line_8, box_per_line_8),
        )

        tensor_16 = np.reshape(
            tensor_16,
            (1, tensor_shape1_stride, box_per_line_16, box_per_line_16),
        )

        tensor_32 = np.reshape(
            tensor_32,
            (1, tensor_shape1_stride, box_per_line_32, box_per_line_32),
        )

        tensor_8 = np.transpose(tensor_8, (0, 2, 3, 1))
        tensor_16 = np.transpose(tensor_16, (0, 2, 3, 1))
        tensor_32 = np.transpose(tensor_32, (0, 2, 3, 1))

        boxes_8 = self.get_boxes(
            tensor_out, tensor_8[..., 64:], reliability_threshold, 8
        )
        boxes_16 = self.get_boxes(
            tensor_out, tensor_16[..., 64:], reliability_threshold, 16
        )
        boxes_32 = self.get_boxes(
            tensor_out, tensor_32[..., 64:], reliability_threshold, 32
        )

        boxes = self.nms_sorted_bboxes(
            boxes_8 + boxes_16 + boxes_32, self.nms_threshold
        )
        return self.scale_boxes(boxes)


class YOLO11_OBB(YOLO11_DET):
    results: List[YOLO_RESULT_OBB]
    _result_type = YOLO_RESULT_OBB
    model_img_size = 1024

    def get_result(self) -> List[YOLO_RESULT_OBB]:
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> List[YOLO_RESULT_OBB]:
        return super().run(img, reliability_threshold, nms_threshold)

    def post_process(self, reliability_threshold):
        boxes = super().post_process(reliability_threshold)
        tensor_angle = self.npu.output_buffer.get(4)

        for i in boxes:
            i.angle = (tensor_angle[i.index_in_all_boxes] - 0.25) * np.pi
        return boxes


class YOLO11_SEG(YOLO11_DET):
    results: List[YOLO_RESULT_SEG]
    _result_type = YOLO_RESULT_SEG
    model_img_size = 640

    def get_result(self) -> List[YOLO_RESULT_SEG]:
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> List[YOLO_RESULT_SEG]:
        return super().run(img, reliability_threshold, nms_threshold)

    def post_process(self, reliability_threshold):
        # self.npu.save_tensor(".")
        boxes = super().post_process(reliability_threshold)
        masks_in = self.npu.output_buffer.get(4)
        masks_in = np.reshape(masks_in, (32, 8400))
        masks_in = np.transpose(masks_in, (1, 0))
        protos = self.npu.output_buffer.get(5)
        protos = np.reshape(protos, (32, 160 * 160))

        def crop_mask(masks, xyxy):
            h, w = masks.shape
            x1, y1, x2, y2 = xyxy
            r = np.arange(w)[None, :]  # rows shape(1, w)
            c = np.arange(h)[:, None]  # cols shape(h, 1)
            mask = (r >= x1) & (r <= x2) & (c >= y1) & (c <= y2)
            cropped_mask = masks * mask
            return cropped_mask

        for i in boxes:
            box_mask = np.dot(masks_in[i.index_in_all_boxes], protos)
            box_mask = sigmoid(box_mask)
            box_mask = np.reshape(box_mask, (160, 160))
            xyxy = [
                int((i.xywh[0] - i.xywh[2] / 2) / 4),
                int((i.xywh[1] - i.xywh[3] / 2) / 4),
                int((i.xywh[0] + i.xywh[2] / 2) / 4),
                int((i.xywh[1] + i.xywh[3] / 2) / 4),
            ]

            box_mask = crop_mask(box_mask, xyxy)
            i._raw_mask = box_mask
            mask = box_mask > reliability_threshold
            indices = np.argwhere(mask)
            img = np.zeros((160, 160, 1), dtype=np.uint8)
            img[indices[:, 0], indices[:, 1], :] = 255
            img = self.resize_model2img(img)
            # img = cv2.GaussianBlur(img, (3, 3), 0) # 对img模糊处理，降低边缘锯齿
            i.mask = img

            contours, hierarchy = cv2.findContours(
                img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = np.vstack(contours).squeeze().tolist()
            i.contours = contours
        return boxes


class YOLO11_POSE(YOLO11_DET):
    results: List[YOLO_RESULT_POSE]
    _result_type = YOLO_RESULT_POSE
    model_img_size = 640

    def get_result(self) -> List[YOLO_RESULT_POSE]:
        return super().get_result()

    def run(
        self, img, reliability_threshold=0.5, nms_threshold=0.5
    ) -> List[YOLO_RESULT_POSE]:
        return super().run(img, reliability_threshold, nms_threshold)

    def post_process(self, reliability_threshold):
        # self.npu.save_tensor(".")
        boxes = super().post_process(reliability_threshold)

        tensor_predkpt = self.npu.output_buffer.get(4)
        tensor_predkpt = np.reshape(tensor_predkpt, (-1, 8400))
        tensor_predkpt = np.transpose(tensor_predkpt, (1, 0))

        tensor_kpt = self.npu.output_buffer.get(5)
        tensor_kpt = np.reshape(tensor_kpt, (-1, 8400))
        tensor_kpt = np.transpose(tensor_kpt, (1, 0))

        original_height, original_width, _ = self.img_raw.shape
        scale = min(
            self.model_img_size / original_width, self.model_img_size / original_height
        )
        pad_x = (self.model_img_size - original_width * scale) / 2
        pad_y = (self.model_img_size - original_height * scale) / 2

        for box in boxes:
            box_predkpt = tensor_predkpt[box.index_in_all_boxes]
            box_predkpt = np.reshape(box_predkpt, (-1, 3))

            box_kpt = tensor_kpt[box.index_in_all_boxes]
            box_kpt = np.reshape(box_kpt, (-1, 3))
            box_predkpt[..., 2] = sigmoid(box_kpt[..., 2])
            box.keypoints = []
            for i in box_predkpt:
                kpt = _YOLO_KEYPOINT()
                kpt.xy = (int((i[0] - pad_x) / scale), int((i[1] - pad_y) / scale))
                kpt.visibility = i[2]
                box.keypoints.append(kpt)
        return boxes
