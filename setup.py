from setuptools import find_packages, setup

install_requires = [
    "gym",
    "numpy",
    "opencv-python",
    "matplotlib",
    "seaborn",
    "pandas",
    "pathlib",
    "tqdm",
    "cpprb",
    "torch",
]

extras_require = {
    "tests": ["pytest", "pytest-benchmark", "pysen[lint]"],
    "clearml": [
        "clearml",
        "clearml-agent",
    ],
    "experiments": ["celluloid", "pybullet"],
}

setup(
    name="ShinRL",
    version="0.0.1",
    description=("A python library for debugging reinforcement learning"),
    packages=find_packages(),
    url="https://github.com/syuntoku14/ShinRL",
    author="Toshinori Kitamura",
    author_email="syuntoku14@gmail.com",
    install_requires=install_requires,
    extras_require=extras_require,
)
