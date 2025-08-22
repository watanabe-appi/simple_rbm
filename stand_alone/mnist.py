import os
import argparse
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.append(base_dir)

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
