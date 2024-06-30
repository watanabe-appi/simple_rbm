try:
    import cupy as np

    has_GPU = True
except ImportError:
    import numpy as np

    has_GPU = False
import logging
from typing import Dict


class RBM:
    def __init__(self, visible_num, hidden_num, learning_rate=0.0001, momentum=0.95, seed=0):
        np.random.seed(seed)
        self.visible_num = visible_num
        self.hidden_num = hidden_num

        self.w = np.array(self.xavier_init(
            visible_num, hidden_num), dtype=np.float32)
        self.visible_bias = np.zeros(visible_num, dtype=np.float32)
        self.hidden_bias = np.zeros(hidden_num, dtype=np.float32)

        self.delta_w = np.zeros(
            [self.visible_num, self.hidden_num], dtype=np.float32)
        self.delta_visible_bias = np.zeros(visible_num, dtype=np.float32)
        self.delta_hidden_bias = np.zeros(hidden_num, dtype=np.float32)

        self.learning_rate = learning_rate
        self.momentum = momentum

        self.logger = logging.getLogger(self.__class__.__name__)

    def xavier_init(self, fan_in, fan_out, const=1.0, dtype=np.float32):
        k = const * np.sqrt(6.0 / (fan_in + fan_out))

        return np.random.uniform(-k, k, size=(fan_in, fan_out)).astype(dtype)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def heaviside(self, x):
        y = np.zeros_like(x)
        y[x > 0] = 1
        y[x <= 0] = 0

        return y

    def sample(self, expected_values):
        r = np.random.random(expected_values.shape)
        s = expected_values - r
        sampled_values = self.heaviside(s)

        return sampled_values

    def apply_momentum(self, former_delta_w, new_grad):
        momentum = self.momentum
        learning_rate = self.learning_rate

        return (former_delta_w * momentum) + ((1 - momentum) * learning_rate * new_grad)

    def step(self, batch):
        sampled_hidden = self.sample(self.sigmoid(
            batch.dot(self.w) + self.hidden_bias))
        sampled_visible = self.sample(
            self.sigmoid(sampled_hidden.dot(
                self.w.transpose()) + self.visible_bias)
        )
        re_sampled_hidden = self.sample(
            self.sigmoid(sampled_visible.dot(self.w) + self.hidden_bias)
        )

        positive_grad = batch.transpose().dot(sampled_hidden)
        negative_grad = sampled_visible.transpose().dot(re_sampled_hidden)
        new_grad = positive_grad - negative_grad

        self.delta_w = self.apply_momentum(self.delta_w, new_grad)
        self.delta_visible_bias = self.apply_momentum(
            self.delta_visible_bias, np.mean(batch - sampled_visible, axis=0)
        )
        self.delta_hidden_bias = self.apply_momentum(
            self.delta_hidden_bias, np.mean(
                sampled_hidden - re_sampled_hidden, axis=0)
        )

        self.w += self.delta_w
        self.visible_bias += self.delta_visible_bias
        self.hidden_bias += self.delta_hidden_bias

        error = np.mean(np.square(batch - sampled_visible))

        return error

    def fit(self, data_set, *, epochs=10, batch_size=1):
        if has_GPU:
            self.logger.info("A GPGPU has been detected, so it will be used.")
        else:
            self.logger.info(
                "GPGPUs were not detected, so the computation will proceed with the CPU."
            )
        for epoch in range(epochs):
            self.logger.info("epoch: %d", epoch)
            epoch_error = 0

            data_set = np.array(data_set)
            np.random.shuffle(data_set)
            batch_data = np.array_split(data_set, len(data_set) // batch_size)
            batch_data_num = len(batch_data)

            for batch in batch_data:
                batch_error = self.step(batch)
                epoch_error += batch_error

            self.logger.info("mean squared error: %f",
                             epoch_error / batch_data_num)

    def get_state(self) -> Dict[str, np.ndarray]:
        if has_GPU:
            return {
                "w": self.w.get(),
                "visible_bias": self.visible_bias.get(),
                "hidden_bias": self.hidden_bias.get(),
                "delta_w": self.delta_w.get(),
                "delta_visible_bias": self.delta_visible_bias.get(),
                "delta_hidden_bias": self.delta_hidden_bias.get(),
            }
        else:
            return {
                "w": self.w,
                "visible_bias": self.visible_bias,
                "hidden_bias": self.hidden_bias,
                "delta_w": self.delta_w,
                "delta_visible_bias": self.delta_visible_bias,
                "delta_hidden_bias": self.delta_hidden_bias,
            }

    def set_state(self, state: Dict[str, np.ndarray]):
        self.w = np.array(state["w"])
        self.visible_bias = np.array(state["visible_bias"])
        self.hidden_bias = np.array(state["hidden_bias"])
        self.delta_w = np.array(state["delta_w"])
        self.delta_visible_bias = np.array(state["delta_visible_bias"])
        self.delta_hidden_bias = np.array(state["delta_hidden_bias"])

    def reconstruct(self, input):
        input = np.array(input)
        sampled_hidden = self.sigmoid(input.dot(self.w) + self.hidden_bias)
        sampled_visible = self.sigmoid(
            sampled_hidden.dot(self.w.transpose()) + self.visible_bias
        )
        if has_GPU:
            sampled_visible = sampled_visible.get()
        return sampled_visible

    def get_hidden(self, input):
        hidden = self.sigmoid(input.dot(self.w) + self.hidden_bias)
        if has_GPU:
            hidden = hidden.get()
        return hidden

    def sample_hidden(self, input_visible):
        input_visible = np.array(input_visible)
        sampled_values = self.sample(
            self.sigmoid(input_visible.dot(self.w) + self.hidden_bias)
        )
        if has_GPU:
            sampled_values = sampled_values.get()
        return sampled_values

    def sample_visible(self, input_hidden):
        input_hidden = np.array(input_hidden)
        sampled_values = self.sample(
            self.sigmoid(input_hidden.dot(
                self.w.transpose()) + self.visible_bias)
        )
        if has_GPU:
            sampled_values = sampled_values.get()
        return sampled_values

    def expect_hidden(self, input_visible):
        input_visible = np.array(input_visible)
        expected_values = self.sigmoid(
            input_visible.dot(self.w) + self.hidden_bias)
        if has_GPU:
            expected_values = expected_values.get()
        return expected_values

    def expect_visible(self, input_hidden):
        input_hidden = np.array(input_hidden)
        expected_values = self.sigmoid(input_hidden.dot(
            self.w.transpose()) + self.visible_bias
        )
        if has_GPU:
            expected_values = expected_values.get()
        return expected_values

    def calculate_energy(self, visible, hidden):
        energy = (visible.dot(self.w)).dot(hidden.transpose()) + self.visible_bias.dot(
            visible.transpose()) + self.hidden_bias.dot(hidden.transpose())

        return - energy[0][0]
