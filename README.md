# Simple implementation of Restricted Boltzmann Machine (RBM)

A tiny, educational implementation of a Restricted Boltzmann Machine (RBM).

## Features

* Implemented with NumPy only.
* Optional GPU acceleration: seamlessly switches to CuPy when available (--use-gpu in examples).
* Clean, readable code intended for learning and small experiments.

## Install

### Install from GitHub (recommended)

Please run the following commands in the appropriate folder (repository):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
```

On macOS, Python **3.11** is required to use TensorFlow.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
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

If you have CUDA and want GPU acceleration:
* CUDA 11.2: python -m pip install cupy-cuda112
* CUDA 11.8: python -m pip install cupy-cuda118
* CUDA 12.x: python -m pip install cupy-cuda12x

Then run examples with `--use-gpu`.

## Quickstart: MNIST Example

### Running the Example

```sh
git clone https://github.com/watanabe-appi/simple_rbm.git
cd simple_rbm
source .venv/bin/activate
python -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
python3 -m pip install tensorflow Pillow

# run (CPU)
cd examples
python3 mnist.py

# run (GPU, if CuPy is installed)
python3 mnist.py --use-gpu
```

The script `mnist.py` will:
* Load the MNIST dataset and train an RBM model.
* After training, read 10 sample MNIST digit images (`input_0.png` to `input_9.png`) and generate their reconstructed versions (`output_0.png` to `output_9.png`).
* Save the training loss for each epoch to `loss.dat`.
* Save the trained model parameters in Pickle format as `rbm_mnist.pkl`.

You can also test reconstruction with the trained model using `load_test.py`.
Running:
```sh
python3 load_test.py
```
will load the saved weights from `rbm_mnist.pkl` and, just like `mnist.py`, take `input_0.png` through `input_9.png` and produce reconstructed images `output_0.png` through `output_9.png`.

### Example Output

When you run `mnist.py`, you will see output similar to the following:
```sh
$ python3 mnist.py
# Computation will proceed on the CPU.
Epoch [1/10], KL Divergence: 0.3689
Epoch [2/10], KL Divergence: 0.2504
Epoch [3/10], KL Divergence: 0.2144
Epoch [4/10], KL Divergence: 0.1982
Epoch [5/10], KL Divergence: 0.1875
Epoch [6/10], KL Divergence: 0.1797
Epoch [7/10], KL Divergence: 0.1736
Epoch [8/10], KL Divergence: 0.1685
Epoch [9/10], KL Divergence: 0.1645
Epoch [10/10], KL Divergence: 0.1612
Loss history saved to loss.dat
```

After training, the script will output pairs of images named `input_*.png` and `output_*.png`.  
* `input_*.png`: the original MNIST digit images used as input to the RBM.  
* `output_*.png`: the reconstructed images produced by the trained RBM.  

Below is an example of the output:

| Input                           | Output                            |
| ------------------------------- | --------------------------------- |
| ![input\_0](images/input_0.png) | ![output\_0](images/output_0.png) |
| ![input\_1](images/input_1.png) | ![output\_1](images/output_1.png) |
| ![input\_2](images/input_2.png) | ![output\_2](images/output_2.png) |
| ![input\_3](images/input_3.png) | ![output\_3](images/output_3.png) |
| ![input\_4](images/input_4.png) | ![output\_4](images/output_4.png) |
| ![input\_5](images/input_5.png) | ![output\_5](images/output_5.png) |
| ![input\_6](images/input_6.png) | ![output\_6](images/output_6.png) |
| ![input\_7](images/input_7.png) | ![output\_7](images/output_7.png) |
| ![input\_8](images/input_8.png) | ![output\_8](images/output_8.png) |
| ![input\_9](images/input_9.png) | ![output\_9](images/output_9.png) |

### Running on ISSP System C (University of Tokyo)

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
