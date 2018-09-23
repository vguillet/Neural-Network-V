import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Define a Network class, containing all the methods for train ing and evaluating the network
class Network:
    def __init__(self, inputs, desired_outputs, layer1neurons):
        self.inputs = inputs  # inputs as an array, dimensions are Inputs x Training lines
        self.desired_outputs = desired_outputs  # desired outputs encoded so that a single bit is 1
        self.weightset1 = np.random.rand(self.inputs.shape[1], layer1neurons)  # weights from inputs to hidden layer
        self.weightset2 = np.random.rand(layer1neurons, len(desired_outputs[0]))  # weights from hidden layer to output
        self.output = np.zeros(self.desired_outputs.shape)  # output used for error calculation
        self.mse_lst = []

    # define feed forward function
    def feedforward(self):
        self.hiddenlayer1 = sigmoid(np.dot(self.inputs, self.weightset1))   # feedforward inputs to first layer
        self.output = sigmoid(np.dot(self.hiddenlayer1, self.weightset2))   # feedforward first layer to output

    # define the back propagation method and update the weights
    def backpropagate(self, learning_rate):
        # calculating the change of the weights, calculating first and adding later since the calculation
        # for every layer has to be done first
        weights2_change = np.dot(self.hiddenlayer1.T, (learning_rate * (self.desired_outputs - self.output) * sigmoid_derivative(self.output)))
        weights1_change = np.dot(self.inputs.T, (
                    np.dot(learning_rate * (self.desired_outputs - self.output) * sigmoid_derivative(self.output),
                           self.weightset2.T) * sigmoid_derivative(self.hiddenlayer1)))
        self.weightset1 += weights1_change
        self.weightset2 += weights2_change

    # define the Mean squared error calculations, to be performed at every epoch and to evaluate the network performance
    def mse(self, results, target_lst):
        output = []
        for i in range(len(results)):
            for j in range(len(results[i])):
                if j + 1 == int(target_lst[i][0]):  # check whether the output being checked is the target output
                    results[i][j] = (1 - results[i][j]) ** 2
                else:
                    results[i][j] = (0 - results[i][j]) ** 2

            # record the mean sum of the errors squared and append it to the self.mse list
            output.append(sum(results[i]) / len(results[i]))
        self.mse_lst.append(sum(output) / len(output))
