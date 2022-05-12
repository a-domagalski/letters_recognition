import random
import numpy as np
from math import exp


def sigmoid(x):
    if abs(x) >= 710:
        return 0
    return 1 / (1 - exp(x * (-1)))


class NeuralNetwork:

    def __init__(self, input_size, output_size, learning_rate, max_error, bias=0):
        self.weights = []
        self.input_size = input_size
        self.output_size = output_size
        self.output_layer = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.max_error = max_error
        self.bias = bias
        self.randomize_weights()

    def randomize_weights(self):
        for idx in range(self.input_size):
            self.weights.append([random.uniform(-1, 1) for idx in range(self.output_size)])

    def compute_outputs(self, input_layer):
        output_value_holder = 0
        for out_idx in range(self.output_size):
            for in_idx in range(len(input_layer)):
                output_value_holder += input_layer[in_idx] * self.weights[in_idx][out_idx]
            self.output_layer[out_idx] = sigmoid(output_value_holder) + self.bias

    def propagate_input(self, input_layer, out_target):
        self.compute_outputs(input_layer)
        self.update_weights(input_layer, out_target)

    def update_weights(self, input_layer, out_target):
        for out_idx in range(self.output_size):
            for in_idx in range(len(input_layer)):
                outp = self.output_layer[out_idx]
                basic_error = out_target - outp
                self.weights[in_idx][out_idx] = self.weights[in_idx][out_idx] + self.learning_rate * basic_error * (
                        1 - outp) * input_layer[in_idx] * outp

    def compute_error(self, out_target):
        error = 0
        idx = 0
        for outp in self.output_layer:
            error += pow(out_target - outp, 2) / 2
            idx += 1
        return error


inputs = [11111100011000111111100011000110001,
          11111100001000010000100001000011111]
out_targets = [0, 1]  # 0 - A, 1 - C
split_inputs = []

for inp in inputs:
    split_inputs.append([int(digit) for digit in str(inp)])
net = NeuralNetwork(len(split_inputs[0]), 1, 0.5, 0.01)

error = 10
while error > net.max_error:
    error = 0
    if net.learning_rate < 0.001:
        net.learning_rate = 0.5
    net = NeuralNetwork(len(split_inputs[0]), 1, net.learning_rate, 0.001)
    net.learning_rate *= 0.9
    idx = 0
    for s_input in split_inputs:
        net.propagate_input(s_input, out_targets[idx])
        error += net.compute_error(out_targets[idx])
        idx += 1
        # print(error)

print(f"learning rate {net.learning_rate}")
print(f"error {error}")
print("A - 0", "C - 1")
for s_input in split_inputs:
    net.compute_outputs(s_input)
    print(net.output_layer)
