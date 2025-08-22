import tensorflow as tf
from PIL import Image
import pickle
import numpy as np
import sys
import os

from pathlib import Path
import argparse
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root)) 

from simple_rbm import RBM
from examples.mnist import do_fit



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


def main():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32)
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32)

    rbm = RBM(visible_num=28 * 28, hidden_num=64)
    filename = "rbm_mnist.pkl"
    with open(filename, mode="rb") as f:
        state = pickle.load(f)
    print(f"Model {filename} was loaded.")
    rbm.set_state(state)

    for i in range(10):
        save_img(f"input_{i}.png", x_test[i])
        output = rbm.reconstruct(x_test[i].reshape(1, 28 * 28))[0]
        save_img(f"output_{i}.png", output)
        print(f"input_{i}.png -> RBM -> output_{i}.png")


if __name__ == "__main__":
    main()
