"""main math functions"""

import os
import subprocess
from distutils.spawn import find_executable

from setuptools import find_packages, setup

#########################################
NAME = "plmnist"
VERSION = "0.0.1"
ISRELEASED = True
if ISRELEASED:
    __version__ = VERSION
else:
    __version__ = VERSION + ".dev0"
#########################################


def proto_procedure():
    # Find the Protocol Compiler and compile protocol buffers
    # Only compile if a protocompiler is found, otherwise don't do anything
    if "PROTOC" in os.environ and os.path.exists(os.environ["PROTOC"]):
        protoc = os.environ["PROTOC"]
    elif os.path.exists("../src/protoc"):
        protoc = "../src/protoc"
    elif os.path.exists("../src/protoc.exe"):
        protoc = "../src/protoc.exe"
    elif os.path.exists("../vsprojects/Debug/protoc.exe"):
        protoc = "../vsprojects/Debug/protoc.exe"
    elif os.path.exists("../vsprojects/Release/protoc.exe"):
        protoc = "../vsprojects/Release/protoc.exe"
    else:
        protoc = find_executable("protoc")
        if protoc is None:
            protoc = find_executable("protoc.exe")

    if protoc is not None:
        compile_proto(protoc)


def compile_proto(protoc):
    protoc_command = [
        protoc,
        "-I=/install_custom_package/plmnist/",
        "--python_out=/install_custom_package/plmnist/",
        "compound.proto",
    ]
    subprocess.call(protoc_command)


if __name__ == "__main__":
    proto_procedure()

    setup(
        name=NAME,
        version=__version__,
        description=__doc__.split("\n"),
        long_description=__doc__,
        author="",
        author_email="",
        url="https://github.com/pace-gatech/signac_pytorch_pl_mnist_example",
        download_url="https://github.com/pace-gatech/signac_pytorch_pl_mnist_example/tarball/{}".format(
            __version__
        ),
        packages=find_packages(),
        package_data={
            "/signac_pytorch_pl_mnist_example/": [
            ]
        },
        package_dir={"/signac_pytorch_pl_mnist_example": "/signac_pytorch_pl_mnist_example"},
        include_package_data=True,
        license="MIT",
        zip_safe=False,
        keywords="/signac_pytorch_pl_mnist_example/",
        classifiers=[
            "Development Status :: 0 - 1",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT Open Source License",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Tutorial",
            "Operating System :: Unix",
        ],
    )
