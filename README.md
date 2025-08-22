# Simple implementation of Restricted Boltzmann Machine (RBM)

A tiny, educational implementation of a Restricted Boltzmann Machine (RBM).

## Features

* Implemented with NumPy only.
* Optional GPU acceleration: seamlessly switches to CuPy when available (--use-gpu in examples).
* Clean, readable code intended for learning and small experiments.

## Install

### Install from GitHub (recommended)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# install the library from GitHub via HTTPS
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
```

### Local editable install

```sh
git clone https://github.com/watanabe-appi/simple_rbm.git
cd simple_rbm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### GPU Acceleration (Optional, via CuPy)

if you have CUDA and want GPU acceleration:
* CUDA 11.2: python -m pip install cupy-cuda112
* CUDA 11.8: python -m pip install cupy-cuda118
* CUDA 12.x: python -m pip install cupy-cuda12x

Then run examples with `--use-gpu`.

## Quickstart: MNIST Example

```sh
# clone (if not already)
git clone https://github.com/watanabe-appi/simple_rbm.git
cd simple_rbm
source .venv/bin/activate

# (optional) install example deps
python3 -m pip install tensorflow Pillow

# run (CPU)
cd examples
python3 mnist.py

# run (GPU, if CuPy is installed)
python3 mnist.py --use-gpu
```

### macOS Notes

On macOS, Python **3.11** is required to use TensorFlow.

```sh
git clone https://github.com/watanabe-appi/simple_rbm.git
cd simple_rbm
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
python3 -m pip install tensorflow Pillow

cd examples
python mnist.py
```

### ISSP System C (University of Tokyo) Example

Setup.

```sh
module purge
module load cuda/11.2
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3 -m venv .venv 
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow Pillow pickles
python3 -m pip install cupy-cuda112
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
```

Run on an ACC node

```sh
qsub -I -q i1accs -l select=1:ncpus=64
module purge
module load cuda/11.2
source .venv/bin/activate
cd examples
python3 mnist.py --use-gpu
```

## License

MIT

## Acknowledgement

If you use this code in academic work, a simple citation in the text (or a software reference in your appendix) would be greatly appreciated.
