import numpy as np
import pickle
from pathlib import Path
from typing import Union, Dict


class RBM:
    def __init__(
        self,
        visible_num,
        hidden_num,
        learning_rate=0.0001,
        momentum=0.95,
        seed=0,
        use_GPU=False,
    ):
        self.np = np
        self.use_GPU = use_GPU
        if self.use_GPU:
            try:
                import cupy as cp

                self.np = cp
            except ImportError:
                raise RuntimeError("CuPy is not installed.")

        self.np.random.seed(seed)
        self.loss_history = {}
        self.visible_num = visible_num
        self.hidden_num = hidden_num

        self.w = self.np.array(self.xavier_init(visible_num, hidden_num))
        self.visible_bias = self.np.zeros(visible_num, dtype=self.np.float32)
        self.hidden_bias = self.np.zeros(hidden_num, dtype=self.np.float32)

        self.delta_w = self.np.zeros(
            [self.visible_num, self.hidden_num], dtype=self.np.float32
        )
        self.delta_visible_bias = self.np.zeros(visible_num, dtype=self.np.float32)
        self.delta_hidden_bias = self.np.zeros(hidden_num, dtype=self.np.float32)

        self.learning_rate = learning_rate
        self.momentum = momentum

    def xavier_init(self, fan_in, fan_out, const=1.0):
        k = const * self.np.sqrt(6.0 / (fan_in + fan_out))

        return self.np.random.uniform(-k, k, size=(fan_in, fan_out)).astype(
            self.np.float32
        )

    def sigmoid(self, x):
        return 1 / (1 + self.np.exp(-x))

    def heaviside(self, x):
        y = self.np.zeros_like(x)
        y[x > 0] = 1
        y[x <= 0] = 0

        return y

    def sample(self, expected_values):
        r = self.np.random.random(expected_values.shape)
        s = expected_values - r
        sampled_values = self.heaviside(s)

        return sampled_values

    def apply_momentum(self, former_delta_w, new_grad):
        momentum = self.momentum
        learning_rate = self.learning_rate

        return (former_delta_w * momentum) + ((1 - momentum) * learning_rate * new_grad)

    def step(self, batch):
        sampled_hidden = self.sample(self.sigmoid(batch.dot(self.w) + self.hidden_bias))
        sampled_visible = self.sample(
            self.sigmoid(sampled_hidden.dot(self.w.transpose()) + self.visible_bias)
        )
        re_sampled_hidden = self.sample(
            self.sigmoid(sampled_visible.dot(self.w) + self.hidden_bias)
        )

        positive_grad = batch.transpose().dot(sampled_hidden)
        negative_grad = sampled_visible.transpose().dot(re_sampled_hidden)
        new_grad = positive_grad - negative_grad

        self.delta_w = self.apply_momentum(self.delta_w, new_grad)
        self.delta_visible_bias = self.apply_momentum(
            self.delta_visible_bias, self.np.mean(batch - sampled_visible, axis=0)
        )
        self.delta_hidden_bias = self.apply_momentum(
            self.delta_hidden_bias,
            self.np.mean(sampled_hidden - re_sampled_hidden, axis=0),
        )

        self.w += self.delta_w
        self.visible_bias += self.delta_visible_bias
        self.hidden_bias += self.delta_hidden_bias

        expected_visible = self.sigmoid(
            sampled_hidden.dot(self.w.transpose()) + self.visible_bias
        )

        log_p = self.np.sum(
            self.np.log(
                expected_visible**batch * (1 - expected_visible) ** (1 - batch)
            ),
            axis=1,
        )

        q = self.np.ones(batch.shape[0]) / batch.shape[0]
        log_q = self.np.log(self.np.ones(batch.shape[0]) / batch.shape[0])

        error = self.np.sum(q * (log_q - log_p)) / batch.shape[1]

        return error

    def fit(self, data_set, *, epochs=10, batch_size=1):
        self.loss_history["epoch"] = []
        self.loss_history["kl_divergence"] = []
        if self.use_GPU:
            print("# GPU usage has been enabled. Computation will proceed on the GPU.")
        else:
            print("# Computation will proceed on the CPU.")
        for epoch in range(epochs):
            epoch_error = 0
            data_set = self.np.array(data_set)
            self.np.random.shuffle(data_set)
            batch_data = self.np.array_split(data_set, len(data_set) // batch_size)
            batch_data_num = len(batch_data)

            for batch in batch_data:
                batch_error = self.step(batch)
                epoch_error += batch_error
            divergence = epoch_error / batch_data_num
            self.loss_history["epoch"].append(epoch + 1)
            self.loss_history["kl_divergence"].append(divergence)
            print(f"Epoch [{epoch+1}/{epochs}], KL Divergence: {divergence:.4f}")

    def get_state(self) -> Dict[str, np.ndarray]:
        if self.use_GPU:
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
        self.w = self.np.array(state["w"])
        self.visible_bias = self.np.array(state["visible_bias"])
        self.hidden_bias = self.np.array(state["hidden_bias"])
        self.delta_w = self.np.array(state["delta_w"])
        self.delta_visible_bias = self.np.array(state["delta_visible_bias"])
        self.delta_hidden_bias = self.np.array(state["delta_hidden_bias"])

    def reconstruct(self, input):
        input = self.np.array(input)
        sampled_hidden = self.sigmoid(input.dot(self.w) + self.hidden_bias)
        sampled_visible = self.sigmoid(
            sampled_hidden.dot(self.w.transpose()) + self.visible_bias
        )
        if self.use_GPU:
            sampled_visible = sampled_visible.get()
        return sampled_visible

    def get_hidden(self, input):
        hidden = self.sigmoid(input.dot(self.w) + self.hidden_bias)
        if self.use_GPU:
            hidden = hidden.get()
        return hidden

    def sample_hidden(self, input_visible):
        input_visible = self.np.array(input_visible)
        sampled_values = self.sample(
            self.sigmoid(input_visible.dot(self.w) + self.hidden_bias)
        )
        if self.use_GPU:
            sampled_values = sampled_values.get()
        return sampled_values

    def sample_visible(self, input_hidden):
        input_hidden = self.np.array(input_hidden)
        sampled_values = self.sample(
            self.sigmoid(input_hidden.dot(self.w.transpose()) + self.visible_bias)
        )
        if self.use_GPU:
            sampled_values = sampled_values.get()
        return sampled_values

    def expect_hidden(self, input_visible):
        input_visible = self.np.array(input_visible)
        expected_values = self.sigmoid(input_visible.dot(self.w) + self.hidden_bias)
        if self.use_GPU:
            expected_values = expected_values.get()
        return expected_values

    def expect_visible(self, input_hidden):
        input_hidden = self.np.array(input_hidden)
        expected_values = self.sigmoid(
            input_hidden.dot(self.w.transpose()) + self.visible_bias
        )
        if self.use_GPU:
            expected_values = expected_values.get()
        return expected_values

    def calculate_energy(self, visible, hidden):
        visible = self.np.array(visible)
        hidden = self.np.array(hidden)
        energy = (
            (visible.dot(self.w)).dot(hidden.transpose())
            + self.visible_bias.dot(visible.transpose())
            + self.hidden_bias.dot(hidden.transpose())
        )

        return -energy[0][0]

    def save(self, filename: Union[str, Path]) -> None:
        path = Path(filename)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        state = self.get_state()
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model state saved to {path}.")

    def load(self, filename: Union[str, Path]) -> "RBM":
        path = Path(filename)
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.set_state(state)
        print(f"Model state loaded from {path}.")
        return self

    def save_loss(self, filename: str = "loss.dat") -> None:
        path = Path(filename)
        with open(path, "w") as f:
            f.write("# epoch KL_divergence\n")
            for e, d in zip(
                self.loss_history["epoch"], self.loss_history["kl_divergence"]
            ):
                f.write(f"{e} {d}\n")
        print(f"Loss history saved to {path}")
