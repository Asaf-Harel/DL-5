import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from unit10 import c2w1_init_utils as u10
from DL2 import *


def start(n):
    print(f"-------------------- Exercise {n} --------------------")


def end():
    print("-------------------------------------------------------\n")


# -------------------- Exercise 1.1 --------------------
start(1.1)

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()

np.random.seed(1)
hidden1 = DLLayer("Perseptrons 1", 30, (12288,), "relu", W_initialization="Xavier", learning_rate=0.0075,
                  optimization='adaptive')
hidden2 = DLLayer("Perseptrons 2", 15, (30,), "trim_sigmoid", W_initialization="He", learning_rate=0.1)
print(hidden1)
print(hidden2)

end()

# -------------------- Exercise 1.2 --------------------
start(1.2)

hidden1 = DLLayer("Perceptron 1", 10, (10,), "relu", W_initialization="Xavier", learning_rate=0.0075)
hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
hidden1.save_weights("SaveDir", "Hidden1")
hidden2 = DLLayer("Perceptron 2", 10, (10,), "trim_sigmoid", W_initialization="SaveDir/Hidden1.h5", learning_rate=0.1)
print(hidden1)
print(hidden2)

model = DLModel()
model.add(hidden1)
model.add(hidden2)

dir = "model"
model.save_weights(dir)
print(os.listdir(dir))

end()

# -------------------- Exercise 1.3 --------------------
start(1.3)
hidden1 = DLLayer("Perceptron 1", 10, (2,), "relu", "zeros", 0.01)
hidden2 = DLLayer("Perceptron 2", 5, (10,), "relu", "zeros", 0.01)
output = DLLayer("Perceptron 3", 1, (5,), "trim_sigmoid", "zeros", 1.0)

model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(output)

model.compile("cross entropy", 0.5)
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
axes = plt.gca()
axes.set_ylim([0.65, 0.75])
plt.title("Model with -zeros- initialization")
plt.show()
end()

# -------------------- Exercise 1.4 --------------------
start(1.4)
hidden1 = DLLayer("Perceptron 1", 10, (2,), "relu", learning_rate=0.01)
hidden2 = DLLayer("Perceptron 2", 5, (10,), "relu", learning_rate=0.01)
output = DLLayer("Perceptron 3", 1, (5,), "trim_sigmoid", learning_rate=1.0)

model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(output)

model.compile("cross entropy", 0.5)
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title("–random- initialization")
plt.show()
plt.title("Model with –random- initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)
predictions = model.predict(train_X)
print('Train accuracy: %d' % float((np.dot(train_Y, predictions.T) +
                                    np.dot(1 - train_Y, 1 - predictions.T)) / float(train_Y.size) * 100) + '%')
predictions = model.predict(test_X)
print('Test accuracy: %d' % float((np.dot(test_Y, predictions.T) +
                                   np.dot(1 - test_Y, 1 - predictions.T)) / float(test_Y.size) * 100) + '%')
end()

# -------------------- Exercise 1.5 --------------------
start(1.5)
hidden1 = DLLayer("Perceptron 1", 10, (2,), "relu", "He", 0.01)
hidden2 = DLLayer("Perceptron 2", 5, (10,), "relu", "He", learning_rate=0.01)
output = DLLayer("Perceptron 3", 1, (5,), "trim_sigmoid", "He", learning_rate=1.0)

model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(output)

model.compile("cross entropy", 0.5)
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title("–random- initialization")
plt.show()
plt.title("Model with –random- initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)
predictions = model.predict(train_X)
print('Train accuracy: %d' % float((np.dot(train_Y, predictions.T) +
                                    np.dot(1 - train_Y, 1 - predictions.T)) / float(train_Y.size) * 100) + '%')
predictions = model.predict(test_X)
print('Test accuracy: %d' % float((np.dot(test_Y, predictions.T) +
                                   np.dot(1 - test_Y, 1 - predictions.T)) / float(test_Y.size) * 100) + '%')
end()
