import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from PIL import Image
from simple_rbm import RBM
import numpy as np
import argparse


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
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32)
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32)
    rbm.fit(x_train, epochs=10, batch_size=1000)
    rbm.save_loss()
    rbm.save("rbm_mnist.pkl")


if __name__ == "__main__":
    main()
