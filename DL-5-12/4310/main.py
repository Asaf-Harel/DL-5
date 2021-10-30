import numpy as np
import h5py
import matplotlib.pyplot as plt
from DL7 import *


def start(n):
    print(f'\n-------------------- Exercise {n} --------------------')


plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# -------------- 1 -------------
start(1)

np.random.seed(1)
linear = DLLayer("line", 2, (3,), activation='NoActivation', W_initialization="Xavier", learning_rate=0.001)
linear.init_weights("Xavier")  # Do the Xavier twice...
print(linear)
print(linear.W)
convValid = DLConv("Valid", 3, (3, 15, 20), filter_size=(3, 3), strides=(1, 2), W_initialization="He",
                   padding="Valid", learning_rate=0.01)
print(convValid)

convSame = DLConv("Same", 2, (3, 30, 64), filter_size=(5, 5), strides=(1, 1), W_initialization="He", padding="Same",
                  learning_rate=0.1, optimization='adaptive', regularization="L2")
print(convSame)

conv34 = DLConv("34", 2, (3, 28, 28), filter_size=(2, 2), strides=(1, 1), W_initialization="He", padding=(3, 4),
                learning_rate=0.07, optimization='adaptive', regularization="L2")

print(conv34)
print(conv34.W)

# -------------- 2 -------------
start(2)

np.random.seed(1)
prev_A = np.random.randn(3, 4, 4, 10)
test = DLConv("test forward", 8, (3, 4, 4), filter_size=(2, 2), strides=(2, 2), padding=(2, 2),
              W_initialization="He", activation="no_activation")
A = test.forward_propagation(test, prev_A)
print("A's mean =", np.mean(A))
print("A.shape =", str(A.shape))
print("A[3,2,1] =", A[3, 2, 1])
print("W.shape =", str(test.W.shape))

# -------------- 3 -------------
start(3)

np.random.seed(1)
prev_A = np.random.randn(3, 4, 4, 10)
test = DLConv("test backward", 8, (3, 4, 4), filter_size=(2, 2), strides=(2, 2), padding=(2, 2),
              W_initialization="He")

A = test.forward_propagation(test, prev_A)
dA = A * np.random.randn(8, 4, 4, 10)
dA_prev = test.backward_propagation(test, dA)

print("dA_prev's mean =", np.mean(dA_prev))
print("dA_prev.shape =", str(dA_prev.shape))
print("dA_prev[1,2,3] =", dA_prev[1, 2, 3])
print("dW shape =", test.dW.shape)
print("dW[3,2,1] =", test.dW[3, 2, 1])
print("db = ", test.db)

# -------------- 4 -------------
start(4)

conv_layer = DLConv("test conv layer", 7, (3, 100, 100), learning_rate=0.1, activation="relu", filter_size=(3, 3),
                    strides=(1, 1), W_initialization="He", padding='Same', optimization='adaptive', regularization="L2")

print(conv_layer)

# -------------- 5 -------------
start(5)

np.random.seed(1)

A_prev = np.random.randn(3, 100, 100, 10)

test = DLMaxPooling("test max pooling", (3, 100, 100), filter_size=(3, 3), strides=(2, 2))
print(test)
Z = test.forward_propagation(test, A_prev)
print("Z.shape =", str(Z.shape))
print("Z[1,2,3] =", Z[1, 2, 3])
dZ = Z * np.random.randn(3, 49, 49, 10)
dA_prev = test.backward_propagation(test, dZ)
print("dA_prev's mean =", np.mean(dA_prev))
print("dA_prev.shape =", str(dA_prev.shape))
print("dA_prev[1,2,3] =", dA_prev[1, 2, 3])

# -------------- 6 -------------
start(6)

np.random.seed(1)

check_X = np.random.randn(3, 28, 28, 3)
check_Y = np.random.rand(1, 3) > 0.5

test_conv = DLConv("test conv", 12, (3, 28, 28), learning_rate=0.1, filter_size=(3, 3), padding='Same', strides=(1, 1),
                   activation="sigmoid", W_initialization="He")

test_maxpooling = DLMaxPooling("test maxpool", (12, 28, 28), filter_size=(2, 2), strides=(2, 2))
test_flatten = DLFlatten("test flatten", (12, 14, 14))

test_layer1 = DLLayer("test layer1", 17, (12 * 14 * 14,), activation="tanh", learning_rate=0.1, regularization='L2',
                      W_initialization="He")

test_layer2 = DLLayer("test layer2", 1, (17,), activation="sigmoid", learning_rate=0.1, W_initialization="He")

DNN = DLModel("Test div model")
DNN.add(test_conv)
DNN.add(test_maxpooling)
DNN.add(test_flatten)
DNN.add(test_layer1)
DNN.add(test_layer2)
DNN.compile("squared_means")
print(DNN)
