from src.Network import Network
import numpy as np
import matplotlib.pyplot as plt


# ___________________________________________________________DATA PRE-PROCESSING
# function to normalise an array
def normalize(array):
    array_normalised = np.zeros(np.shape(array))
    for i in range(len(array[0][:])):
        normalised_value = (array[:][i] - min(array[:][i]))/(max(array[:][i]) - min(array[:][i]))
        array_normalised[:][i] = normalised_value
    return array_normalised


# Converting data to an array
# ---------------------------------------------------
# convert features to be used as input to a list
feature_lst = [[float(x) for x in lst] for lst in
               [row.strip().split(",") for row in open("../resources/smallfeatures.txt").read().strip().split("\n")]]

# convert target data to a list
target_lst = [[float(x) for x in lst] for lst in
              [row.strip().split(",") for row in open("../resources/smalltargets.txt").read().strip().replace("\n", ",").split(",")]]

# TODO check what happens here in the data preparation:
with open("../resources/smallfeatures.txt") as features:
    target_lst2 = np.zeros((len(features.readlines()), 7))

i = 0
for row in open("../resources/smalltargets.txt").read().strip().replace("\n", ",").split(","):
    target_lst2[i][int(row)-1] = 1
    i = i + 1

# convert list of lists to array
features_array = np.asarray(feature_lst)
target_array = np.asarray(target_lst2)

# normalize the array obtained using the normalize function
features_array_n = normalize(features_array)


# ___________________________________________________________NETWORK INIT
# Variables initialization
# ---------------------------------------------------
epoch = 1500                                        # setting training iterations
learning_rate = 0.3                                 # setting learning rate
layer1neurons = 8                                   # number of hidden layers neurons
layer2neurons = 8                                   # number of hidden layers neurons
layer3neurons = 8                                   # number of hidden layers neurons

inputlayer_neurons = features_array_n.shape[1]      # number of features in data set
# ---------------------------------------------------

# create the network with inputs, and desired outputs
nn = Network(features_array_n, target_array, layer1neurons, layer2neurons, layer3neurons)

# iterate epoch numbers of times
for i in range(epoch):
    nn.feedforward()                        # run feedforward
    nn.backpropagate(learning_rate)         # run feedback
    nn.mse(nn.output, target_lst)           # compute mse

    print(str(i) + "/" + str(epoch))        # output which epoch the computer is at

# print("output")
# print(nn.output)
# ___________________________________________________________RESULT PROCESSING

print(nn.mse_lst)

# plot the mse per test epoch
plt.plot(range(len(nn.mse_lst)), nn.mse_lst)
plt.show()

