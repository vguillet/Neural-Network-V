import numpy as np


# Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Derivative of Sigmoid Function
def sigmoid_derivative(x):
    return x * (1 - x)