# Example 4.1 - Single Neuron (AND/threshold-style training)
import numpy as np
from numpy import exp, array, random, dot

inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
outputs = array([[0, 1, 1, 1]]).T
random.seed(1)
weights = 2 * random.random((2, 1)) - 1

def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

x = array([1, 0])
print(neuron(x, weights))
