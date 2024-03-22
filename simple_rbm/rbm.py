import numpy as np 
import logging
from typing import Dict


class RBM:
    def __init__(self, visible_num, hidden_num, learning_rate = 0.001, momentum = 0.95):
        self.visible_num = visible_num
        self.hidden_num = hidden_num

        self.w = np.array(self.xavier_init(visible_num, hidden_num), dtype=np.float32)
        self.visible_bias = np.zeros(visible_num, dtype=np.float32)
        self.hidden_bias = np.zeros(hidden_num, dtype=np.float32)

        self.delta_w = np.zeros([self.visible_num, self.hidden_num], dtype=np.float32)
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


    def sample(self, expected_values):
        r = np.random.random(expected_values.shape)
        s = expected_values - r
        sampled_values = np.heaviside(s, 0)

        return sampled_values


    def apply_momentum(self, former_delta_w, new_grad):
        momentum = self.momentum
        learning_rate = self.learning_rate

        return (former_delta_w * momentum) + ((1 - momentum) * learning_rate * new_grad)


    def step(self, batch):
        sampled_hidden = self.sample(self.sigmoid(batch.dot(self.w) + self.hidden_bias))
        sampled_visible = self.sample(self.sigmoid(sampled_hidden.dot(self.w.transpose()) + self.visible_bias))
        re_sampled_hidden = self.sample(self.sigmoid(sampled_visible.dot(self.w) + self.hidden_bias))

        positive_grad = batch.transpose().dot(sampled_hidden)
        negative_grad = sampled_visible.transpose().dot(re_sampled_hidden)
        new_grad = positive_grad - negative_grad

        self.delta_w = self.apply_momentum(self.delta_w, new_grad)
        self.delta_visible_bias = self.apply_momentum(self.delta_visible_bias, np.mean(batch - sampled_visible, axis=0))
        self.delta_hidden_bias = self.apply_momentum(self.delta_hidden_bias, np.mean(sampled_hidden - re_sampled_hidden, axis=0))

        self.w += self.delta_w
        self.visible_bias += self.delta_visible_bias
        self.hidden_bias += self.delta_hidden_bias

        error = np.mean(np.square(batch - sampled_visible))

        return error
    

    def train(self, data_set, epoch_num, batch_size):
        for epoch in range(epoch_num):
            self.logger.info("epoch: %d", epoch)
            epoch_error = 0
            
            np.random.shuffle(data_set)
            batch_data = np.array_split(data_set, len(data_set) // batch_size)
            batch_data_num = len(batch_data)

            for batch in batch_data:
                batch_error = self.step(batch)
                epoch_error += batch_error

            self.logger.info("mean squared error: %f", epoch_error / batch_data_num)

    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "w": self.w,
            "visible_bias": self.visible_bias,
            "hidden_bias": self.hidden_bias,
            "delta_w": self.delta_w,
            "delta_visible_bias": self.delta_visible_bias,
            "delta_hidden_bias": self.delta_hidden_bias
        }


    def set_state(self, state: Dict[str, np.ndarray]):
        self.w = state["w"]
        self.visible_bias = state["visible_bias"]
        self.hidden_bias = state["hidden_bias"]
        self.delta_w = state["delta_w"]
        self.delta_visible_bias = state["delta_visible_bias"]
        self.delta_hidden_bias = state["delta_hidden_bias"]      


    def reconstruct(self, input):
        sampled_hidden = (self.sigmoid(input.dot(self.w) + self.hidden_bias))
        sampled_visible = (self.sigmoid(sampled_hidden.dot(self.w.transpose()) + self.visible_bias))
        return  sampled_visible
    

    def get_hidden(self, input):
        hidden = self.sigmoid(input.dot(self.w) + self.hidden_bias)

        return hidden


   