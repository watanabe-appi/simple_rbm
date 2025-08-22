from pathlib import Path
import argparse
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)) 

from simple_rbm import RBM
from examples.mnist import do_fit


def main():
    parser = argparse.ArgumentParser(
        description="Run MNIST training with optional GPU usage."
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU computation."
    )
    args = parser.parse_args()
    use_GPU = args.use_gpu
    rbm = RBM(visible_num=28 * 28, hidden_num=64, use_GPU=use_GPU)
    do_fit(rbm)


if __name__ == "__main__":
    main()
