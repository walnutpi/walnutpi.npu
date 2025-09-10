from setuptools import setup, Extension, find_packages
import numpy
import os
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
import subprocess


# 运行make编译
class MakefileBuild(build_ext):
    def run(self):
        try:
            print("Running make command...")
            subprocess.check_call(
                ["make"], cwd=os.path.dirname(os.path.abspath(__file__))
            )
            print("Make completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error running make: {e}")
            print(f"Return code: {e.returncode}")
            # 可以添加更多调试信息
            raise
        except FileNotFoundError:
            print("Error: 'make' command not found. Please install build tools.")
            raise
        super().run()


classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

with open("version", "r") as file:
    version_str = file.read().strip()

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="walnutpi_npu",
    version=version_str,
    author="sc-bin",
    author_email="3335447573@qq.com",
    description="A module to control npu on T-527",
    platforms=["manylinux"],
    long_description=open("README_PY.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/sc-bin/",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-contrib-python"
    ],
    cmdclass={
        "build_ext": MakefileBuild,
    },
    ext_modules=[
        Extension(
            "_awnn_lib",
            sources=["py_awnn_lib.c"],
            libraries=["awnn_t527"],
            library_dirs=[
                current_dir,
                os.path.join(current_dir, "lib"),
            ],  # 在当前目录和lib目录查找动态库
            include_dirs=[numpy.get_include(), os.path.join(current_dir, "header")],
            runtime_library_dirs=[
                current_dir,
                os.path.join(current_dir, "lib"),
            ],  # 运行时也在这些目录查找
        )
    ],
    # 将动态库文件包含在Python包中
    package_data={
        "": ["*.so", "lib/*.so"],
    },
    include_package_data=True,
)
