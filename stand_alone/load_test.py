import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)) 

from simple_rbm import RBM
from examples.load_test import load_test



def main():
    rbm = RBM(visible_num=28 * 28, hidden_num=64)
    load_test(rbm)

if __name__ == "__main__":
    main()
