import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from PIL import Image
import pickle
from simple_rbm import RBM
import numpy as np
import argparse


def save_img(filename, data):
    data = np.array(data)
    data = data.reshape((28, 28))
    img = Image.new("L", (28, 28))
    pix = img.load()
    data = data * 255
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[j][i])
    img2 = img.resize((28 * 5, 28 * 5))
    img2.save(filename)


def do_fit(rbm):
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32)
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32)
    rbm.fit(x_train, epochs=10, batch_size=1000)
    rbm.save_loss()

    for i in range(10):
        save_img(f"input_{i}.png", x_test[i])
        output = rbm.reconstruct(x_test[i].reshape(1, 28 * 28))[0]
        save_img(f"output_{i}.png", output)

    params = rbm.get_state()
    with open("rbm_mnist.pkl", mode="wb") as f:
        pickle.dump(params, f)


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
