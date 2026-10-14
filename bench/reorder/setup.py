from setuptools import find_packages, setup

setup(
    name="reordering-bench",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "scipy",
        "spmatkit",
    ],
    extras_require={
        "groot-py": ["pynndescent", "numba"],
        "dtc": ["datasketch"],
        "dtc-gpu": ["cudf", "cugraph", "cupy-cuda12x"],
        "sporder-cg": ["cupy-cuda12x"],
        "sporder-csg": ["cupy-cuda12x"],
        "sporder-hm": ["cupy-cuda12x"],
    },
)
