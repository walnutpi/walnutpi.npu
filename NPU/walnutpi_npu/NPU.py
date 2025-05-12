import _awnn_lib
import os
import time
import numpy

class awnn:
    context_ptr: int

    class __buffer:
        def count(self):
            return _awnn_lib.get_output_buffer_count(
                self.context_ptr,
            )

        def get(self, index: int) -> numpy.ndarray:
            return _awnn_lib.awnn_get_output_buffer(self.context_ptr, index)

        def __init__(self, context_ptr):
            self.context_ptr = context_ptr

    model_path: str
    output_buffer: __buffer

    def __init__(self, model_path: str):
        self.model_path = model_path
        _awnn_lib.awnn_init()
        # 判断model_path文件是否存在
        if not os.path.exists(self.model_path):
            raise Exception(f"{model_path} file not exists")
        if not self.model_path.endswith(".nb"):
            raise Exception(f"{model_path} must be a .nb file")
        self.context_ptr = _awnn_lib.awnn_create(self.model_path)
        self.output_buffer = self.__buffer(self.context_ptr)

    def __del__(self):
        _awnn_lib.awnn_destroy(self.context_ptr)
        time.sleep(0.1)
        _awnn_lib.awnn_uninit(self.context_ptr)

    def run(self, input_data: bytearray):
        """运行模型"""
        _awnn_lib.awnn_set_input_buffers(self.context_ptr, input_data)
        _awnn_lib.awnn_run(self.context_ptr)

    def run_async(self, input_data: bytearray):
        """后台运行模型"""
        _awnn_lib.awnn_set_input_buffers(self.context_ptr, input_data)
        _awnn_lib.awnn_run_async(self.context_ptr)

    def is_async_running(self) -> bool:
        """后台运行模型等待"""
        return _awnn_lib.is_awnn_async_running()

    def save_tensor(self, output_path: str):
        """保存输入与输出tensor到路径"""
        _awnn_lib.awnn_dump_io(self.context_ptr, output_path)
