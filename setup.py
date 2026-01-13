from setuptools import setup, find_packages

setup(
    name="ftqec",
    version="0.1.0",
    description="Fractal Tensor Quantum Error Correction - A quantum computing simulator using fractal state engines",
    author="MASSIVEMAGNETICS",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
    ],
)
