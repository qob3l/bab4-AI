from math import exp
from numpy import dot

def activate(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

inputs = [1, 2, 3]
weights = [0.5, 0.5, 0.5]

hasil = activate(inputs, weights)
print(hasil)