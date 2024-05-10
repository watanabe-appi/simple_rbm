import tensorflow as tf
import logging
from PIL import Image
import pickle
import sys
import os

try:
    import cupy as np

    has_GPU = True
    print("A GPGPU has been detected, so it will be used.")
except ImportError:
    import numpy as np

    has_GPU = False
    print("GPGPUs were not detected, so the computation will proceed with the CPU.")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from simple_rbm import RBM


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
    logging.basicConfig(level=logging.INFO)
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32)
    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32)

    rbm = RBM(visible_num=28 * 28, hidden_num=64)
    rbm.fit(x_train, epochs=10, batch_size=1000)

    save_img("input.png", x_test[0])
    output = rbm.reconstruct(x_test[:1])[0]
    save_img("output.png", output)

    for i in range(10):
        save_img(f"input_{i}.png", x_test[i])
        output = rbm.reconstruct(x_test[i].reshape(1, 28 * 28))[0]
        save_img(f"output_{i}.png", output)

    params = rbm.get_state()
    with open("params_test.pkl", mode="wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    main()
