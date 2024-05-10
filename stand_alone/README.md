# ローカル動作によるデバッグ用フォルダ

## 概要

通常はこのリポジトリをpip installして利用することが想定されているが、デバッグ時に不便であるため、ローカルのsimple_rbm/rbm.pyを利用して実行するためのフォルダ。

## 利用法

```sh
git clone git@github.com:watanabe-appi/simple_rbm.git 
cd simple_rbm
python3 -m venv myenv 
source myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow Pillow pickles
cd stand_alone
python3 mnist.py
```

## 物性研での利用法

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
cd stand_alone
qsub -I -q i1accs -l select=1:ncpus=64
python3 mnist.py
```