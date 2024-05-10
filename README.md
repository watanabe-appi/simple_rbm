# Simple implementation of Restricted Boltzmann Machine (RBM)

## Test this repository

```sh
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3 -m venv myenv 
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
cd examples
python3 -m pip install tensorflow Pillow pickles
python3 mnist.py
```

## Install

```sh
python3 -m venv myenv 
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install git+ssh://git@github.com/watanabe-appi/simple_rbm.git
```

## Use on ISSP System C

Setup.

```sh
module purge
module load cuda/11.2
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3 -m venv myenv 
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow Pillow pickles
python3 -m pip install cupy-cuda112
```

Execute on an ACC node.

```sh
qsub -I -q i1accs -l select=1:ncpus=64
module purge
module load cuda/11.2
source myenv/bin/activate
cd stand_alone
python3 mnist.py
```

## License

MIT
