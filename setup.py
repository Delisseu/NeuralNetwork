from setuptools import setup, find_packages

setup(
    name="neuralnet",
    version="1.0.0",
    description="Custom neural network library on CuPy",
    author="Delisseu",
    packages=find_packages(),
    install_requires=[
        # "cupy-cudaxx", Нужно выбрать под свою версию CUDA
        "numpy"
    ],
)
