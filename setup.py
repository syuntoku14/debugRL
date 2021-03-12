import os
from setuptools import setup, find_packages


setup(
    name="ShinRL",
    version="0.0.1",
    description=(
        "A python library for debugging reinforcement learning"),
    packages=find_packages(),
    url="https://github.com/syuntoku14/ShinRL",
    author="Toshinori Kitamura",
    author_email="syuntoku14@gmail.com",
    install_requires=[
        "gym",
        "numpy",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "pandas",
        "autopep8",
        "pytest",
        "pytest-benchmark",
        "pathlib",
        "tqdm",
        "pybullet",
        "cpprb",
        "clearml",
        "clearml-agent",
        "torch",
        "torchvision",
        "celluloid"

    ],
)
