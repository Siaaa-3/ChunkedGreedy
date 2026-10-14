from setuptools import find_packages, setup

setup(
    name="spmm-bench",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "pybind11",
        "spmatkit",
    ],
)
