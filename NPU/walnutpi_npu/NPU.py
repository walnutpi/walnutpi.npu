import os
import time
import numpy
from ctypes import CDLL

# 加载存放在项目路径下的动态库
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.normpath(os.path.join(_current_dir, "../lib"))
CDLL(os.path.join(_lib_dir, "libVIPuser.so"))
CDLL(os.path.join(_lib_dir, "libVIPlite.so"))
import _awnn_lib


class _output_buffer:
    def count(self):
        """获取输出buffer数量"""
        return _awnn_lib.get_output_buffer_count(self._context_ptr)

    def get(self, index: int) -> numpy.ndarray:
        """返回指定buffer的内容"""
        return _awnn_lib.awnn_get_output_buffer(self._context_ptr, index)

    def __init__(self, context_ptr):
        self._context_ptr = context_ptr


class _model_data:
    def shape(self):
        """获取输入buffer的shape"""
        return _awnn_lib.get_input_shape(self._context_ptr)

    def __init__(self, context_ptr, model_path):
        self._context_ptr = context_ptr
        self.path = model_path


class awnn:

    _context_ptr = 0
    model_path: str
    output_buffer: _output_buffer
    model: _model_data

    def __init__(self, model_path: str):
        # 判断model_path文件是否存在
        if not os.path.exists(model_path):
            raise Exception(f"{model_path} file not exists")
        if not model_path.endswith(".nb"):
            raise Exception(f"{model_path} must be a .nb file")

        _awnn_lib.awnn_init()

        self._context_ptr = _awnn_lib.awnn_create(model_path)
        self.output_buffer = _output_buffer(self._context_ptr)
        self.model = _model_data(self._context_ptr, model_path)

    def __del__(self):
        if self._context_ptr == 0:
            return
        _awnn_lib.awnn_destroy(self._context_ptr)
        time.sleep(0.1)
        _awnn_lib.awnn_uninit(self._context_ptr)

    def run(self, input_data: bytearray):
        """运行模型"""
        _awnn_lib.awnn_set_input_buffers(self._context_ptr, input_data)
        _awnn_lib.awnn_run(self._context_ptr)

    def run_async(self, input_data: bytearray):
        """后台运行模型"""
        _awnn_lib.awnn_set_input_buffers(self._context_ptr, input_data)
        _awnn_lib.awnn_run_async(self._context_ptr)

    def is_async_running(self) -> bool:
        """后台运行模型等待"""
        return _awnn_lib.is_awnn_async_running()

    def save_tensor(self, output_path: str):
        """保存输入与输出tensor到路径"""
        _awnn_lib.awnn_dump_io(self._context_ptr, output_path)
