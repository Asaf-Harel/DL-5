import numpy as np
import matplotlib.pyplot as plt
from unit10 import utils as u10
import sklearn
import sklearn.datasets
import scipy.io
from DL4 import *


def start(n):
    print(f'\n-------------------- Exercise {n} --------------------')


# ---------------- 1 ----------------
start(1)

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

l1 = DLLayer("Hidden", 6, (5,), "relu", learning_rate=0.1)
l2 = DLLayer("Output", 1, (6,), "sigmoid", learning_rate=0.1)
print("Default:")
print(l1.is_train)  # _is_compiled
print(l2.is_train)

n = DLModel("Example")
n.add(l1)
n.add(l2)
n.set_train(True)
print("After set to True:")
print(l1.is_train)
print(l2.is_train)

# ---------------- 2 ----------------
start(2)

np.random.seed(1)
ll1 = DLLayer("no regularization", 8, (7,), "tanh", learning_rate=0.1)
ll2 = DLLayer("L2 regularization", 18, (7,), "sigmoid", learning_rate=0.1, regularization="L2")
ll2.L2_lambda = 0.5
ll3 = DLLayer("dropout", 8, (17,), "leaky_relu", learning_rate=0.1, regularization="dropout")
print(f"{ll1}\n{ll2}\n{ll3}")

# ---------------- 3 ----------------
start(3)

np.random.seed(2)
l4 = DLLayer("forward dropout", 2, (4,), activation="no_activation", learning_rate=0.1, regularization="dropout")
prev_A = np.random.randn(4, 5) * 10
A_no_dropout = l4.forward_propagation(prev_A)
print("Input with no dropout:")
print(l4._A_prev)
print("Output with no dropout:")
print(A_no_dropout)
np.random.seed(2)
l4.set_train(True)
A_with_dropout = l4.forward_propagation(prev_A)
print("Input with dropout: (same input as without dropout).")
print(l4._A_prev)
print("Output with dropout:")
print(A_with_dropout)

# ---------------- 4 ----------------
start(4)
np.random.seed(2)
A4 = np.random.randn(15, 4) * 5
l5 = DLLayer("backward dropout", 3, (15,), activation="no_activation", W_initialization="Xavier", learning_rate=0.1,
             regularization="dropout")
l5.set_train(True)
A5_with_dropout = l5.forward_propagation(A4)
dA5 = np.random.randn(3, 4) * 7
dA4 = l5.backward_propagation(dA5)
print("A4 with dropout:")
print(l5._A_prev)
print("dA4:")
print(dA4)

# ---------------- 5 ----------------
start(5)

np.random.seed(2)
l1 = DLLayer("Hidden1", 6, (5,), activation="no_activation", learning_rate=0.1, regularization="L2",
             W_initialization="random")
l2 = DLLayer("Hidden2", 12, (6,), activation="tanh", learning_rate=0.1, regularization="L2", W_initialization="random")
l3 = DLLayer("Output", 1, (12,), activation="sigmoid", learning_rate=0.1, W_initialization="random")

n = DLModel("L2 model")
n.add(l1)
n.add(l2)
n.add(l3)
n.compile("cross_entropy", threshold=0.6)

Y_hat = np.random.rand(1, 17)
Y = np.random.rand(1, 17)
Y = np.where(Y > 0.4, 1, 0)

print(f"Cost with L2 regularization: {n.compute_cost(Y_hat, Y)}")
l1.L2_lambda = 0
l2.L2_lambda = 0
print(f"Cost without L2 regularization:{n.compute_cost(Y_hat, Y)}")

# ---------------- 6 ----------------
start(6)

np.random.seed(2)
l = DLLayer("backward L2", 7, (4,), learning_rate=0.1, activation="no_activation", regularization="L2",
            W_initialization="random")
l.W *= 100
prev_A = np.random.randn(4, 11) * 5
Z = l.forward_propagation(prev_A)
dZ = np.random.randn(*Z.shape)
dA_prev = l.backward_propagation(dZ)
print(f"dW with regularization:\n{l.dW}")
l.L2_lambda = 0
dA_prev = l.backward_propagation(dZ)
print(f"dW with no regularization:\n{l.dW}")

# ---------------- 7 ----------------
start(7)


def print_regularization_cost(model, X, Y, Y_hat):
    s = ""
    m = Y.shape[1]

    for l in model.layers:
        reg_cost = l.regularization_cost(m)
        if reg_cost > 0:
            s += f"\n\t{l.name}: {reg_cost}"

    return s


# ---------------- 8 ----------------
start(8)

np.random.seed(2)
train_X, train_Y, test_X, test_Y = u10.load_2D_dataset()

model = DLModel()
model.add(DLLayer("Layer 1", 64, (train_X.shape[0],), "relu", "He", 0.05, regularization="L2"))
model.add(DLLayer("Layer 2", 32, (64,), "relu", "He", 0.05, regularization="L2"))
model.add(DLLayer("Layer 3", 5, (32,), "relu", "He", 0.05, regularization="L2"))
model.add(DLLayer("Output Layer", 1, (5,), "sigmoid", "He", 0.05))

model.compile("cross entropy")

costs = model.train(train_X, train_Y, 20000)
print("train accuracy:", np.mean((model.forward_propagation(train_X) > 0.7) == train_Y))
print("test accuracy:", np.mean((model.forward_propagation(test_X) > 0.7) == test_Y))
plt.title(f"Model no regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
u10.plot_decision_boundary(model, train_X, train_Y)
u10.print_costs(costs, 20000)
