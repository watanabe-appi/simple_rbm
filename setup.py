from setuptools import setup, find_packages

setup(
    name='simple_rbm',
    version='0.1',
    packages=find_packages(),
    description='Simple Restricted Boltzmann Machine implementation with NumPy',
    long_description='A minimal RBM implementation in Python using NumPy. Supports CuPy for GPU acceleration.',
    author='H. Watanabe',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ],
)

