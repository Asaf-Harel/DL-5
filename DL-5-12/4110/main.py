import numpy as np
import h5py
import matplotlib.pyplot as plt
import unit10.utils as u10
from DL6 import *


def start(n):
    print(f'\n-------------------- Exercise {n} --------------------')


plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# ---------------- 1 ----------------
start(1)

l1 = DLLayer("l1", 2, (3,), learning_rate=0.01, optimization="adam")
l2 = DLLayer("l2", 3, (2,), learning_rate=0.01)
print(l1)
print(l2)

parameters, grads, v, s = u10.update_parameters_with_adam_test_case()
l1.W = parameters["W1"]
l1.b = parameters["b1"]
l1.dW = grads["dW1"]
l1.db = grads["db1"]
l1.adam_v_dW = v["dW1"]
l1.adam_v_db = v["db1"]
l1.adam_s_dW = s["dW1"]
l1.adam_s_db = s["db1"]

l1.update_parameters(2)
print("W1 = " + str(l1.W))
print("b1 = " + str(l1.b))
print(f"v_dW1 = {l1._adam_v_dW}")
print(f"v_db1 = {l1._adam_v_db}")
print(f"s_dW1 = {l1._adam_s_dW}")
print(f"s_db1 = {l1._adam_s_db}")

# ---------------- 2 ----------------
start(2)

np.random.seed(1)
train_X, train_Y = u10.load_moons()
opt = 'adam'

hidden1 = DLLayer("Hidden 1", 64, (2,), "relu", W_initialization="He", learning_rate=0.05, optimization=opt)
hidden2 = DLLayer("Hidden 2", 32, (64,), "relu", W_initialization="He", learning_rate=0.05, optimization=opt)
hidden3 = DLLayer("Hidden 3", 5, (32,), "relu", W_initialization="He", learning_rate=0.05, optimization=opt)
hidden4 = DLLayer("Output", 1, (5,), "sigmoid", W_initialization="He", learning_rate=0.05, optimization=opt)

model = DLModel("Model with mini batches and adam")
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
model.add(hidden4)

model.compile("cross_entropy", 0.5)
costs = model.train(train_X, train_Y, 400, 64)
plt.plot(costs)
plt.show()

train_predict = model.forward_propagation(train_X)
accuracy = np.sum((train_predict > 0.7) == train_Y) / train_X.shape[1]
print("accuracy:", str(accuracy))
plt.title(model.name)
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
u10.plot_decision_boundary(model, train_X, train_Y)
