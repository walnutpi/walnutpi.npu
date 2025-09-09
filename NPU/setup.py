from setuptools import setup, Extension, find_packages
import glob
import datetime
import numpy
import os

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
    ],
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
