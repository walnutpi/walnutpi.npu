from setuptools import setup, Extension, find_packages
import os
from setuptools.command.build_ext import build_ext
import subprocess

# 检查是否在目标平台上
__model_path = "/proc/device-tree/model"
_model = None
if os.path.exists(__model_path):
    with open(__model_path, "r") as f:
        _model = f.read().strip()
    print(f"Model: {_model}")


# 运行make编译的自定义命令
class MakefileBuild(build_ext):
    def run(self):
        # 编译t527的库
        if "walnutpi-2b" in _model:
            try:
                print("Running make command...")
                subprocess.check_call(
                    ["make"],
                    cwd=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "walnutpi_npu/_awnn_lib/t527",
                    ),
                )
                print("Make completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error running make: {e}")
                print(f"Return code: {e.returncode}")
                raise
            except FileNotFoundError:
                print("Error: 'make' command not found. Please install build tools.")
                raise
        else:
            print("Not on target platform, skipping make command...")

        super().run()


def get_numpy_include():
    try:
        import numpy

        return numpy.get_include()
    except ImportError:
        return ""


ext_modules = []
if _model is not None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "walnutpi-2b" in _model:
        numpy_include = get_numpy_include()
        if numpy_include:
            ext_modules = [
                Extension(
                    "_awnn_lib",
                    sources=["walnutpi_npu/_awnn_lib/py_awnn_lib.c"],
                    libraries=["awnn_t527"],
                    library_dirs=[
                        current_dir,
                        os.path.join(current_dir, "walnutpi_npu/_awnn_lib/"),
                        os.path.join(current_dir, "walnutpi_npu/_awnn_lib/lib"),
                    ],
                    include_dirs=[numpy_include, os.path.join(current_dir, "header")],
                    runtime_library_dirs=[
                        current_dir,
                        os.path.join(current_dir, "lib"),
                    ],
                )
            ]

# 项目元数据现在从 pyproject.toml 读取，所以一些东西不在这里定义
setup(
    packages=find_packages(),
    cmdclass={
        "build_ext": MakefileBuild,
    },
    ext_modules=ext_modules,
    package_data={
        "": ["*.so", "lib/*.so"],
    },
    include_package_data=True,
)
