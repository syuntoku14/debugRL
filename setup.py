import os
from setuptools import setup, find_packages


setup(
    name="debugRL",
    version="0.0.1",
    description=(
        "A python library for debugging reinforcement learning"),
    packages=find_packages(),
    url="https://github.com/syuntoku14/debugQ",
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
        "pathlib"
    ],
    extra_requires={
        "solver": [
            "cpprb",
            "clearml",
            "clearml-agent",
            "torch",
            "torchvision"
        ],
    }
)
