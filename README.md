# Simple implementation of Restricted Boltzmann Machine (RBM)

## Install

```sh
python3 -m venv .venv 
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
```

## Test this repository

```sh
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3 -m venv .venv 
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
python3 -m pip install tensorflow Pillow pickles
cd examples
python3 mnist.py
```

## Use on Mac

On MacOSX, you should explicitly use Python 3.11.

```sh
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3.11 -m venv .venv 
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
python3 -m pip install tensorflow Pillow pickles
cd examples
python3 mnist.py
```

## Use on ISSP System C

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

Execute on an ACC node.

```sh
qsub -I -q i1accs -l select=1:ncpus=64
module purge
module load cuda/11.2
source .venv/bin/activate
cd examples
python3 mnist.py
```

## License

MIT
