from setuptools import setup, Extension, find_packages
import glob
import datetime
import numpy

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

with open("version", "r") as file:
    version_str = file.read().strip()

setup(
    name="npu",
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
            library_dirs=["/usr/lib"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
