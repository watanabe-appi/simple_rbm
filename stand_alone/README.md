# Standalone code for debugging

This repository is supposed to be pip installed and used. However, since it is inconvenient for debugging, this folder is used to run using the local simple_rbm/rbm.py.

## Usage

```sh
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3 -m venv .venv 
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow Pillow
cd stand_alone
python3 mnist.py
```
