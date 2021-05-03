import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from unit10 import c1w3_utils as u10
from DL1 import *

np.random.seed(1)
X, Y = u10.load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % m)

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y[0, :])

# Plot the decision boundary for logistic regression
u10.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(
        Y.size) * 100) + '% ' + "(percentage of correctly labelled datapoints)")

# -------------------- Exercise 3.1 --------------------
layer1 = DLLayer("layer 1", 4, (2,), "tanh", "random", 0.1)
layer2 = DLLayer("layer 2", 1, (4,), "sigmoid", "random", 0.1)

model = DLModel()
model.add(layer1)
model.add(layer2)

model.compile("cross_entropy", 0.5)

print(model)

# -------------------- Exercise 3.2 --------------------
costs = model.train(X, Y, 10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

predictions = model.predict(X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) +
                              np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# -------------------- Exercise 3.3 --------------------
layer1 = DLLayer("layer 1", 10, (2,), "relu", "random", 0.01, "adaptive")
layer2 = DLLayer("layer 2", 20, (10,), "tanh", "random", 0.1, "adaptive")
layer3 = DLLayer("layer 3", 1, (20,), "sigmoid", "random", 0.01, "adaptive")

model = DLModel()
model.add(layer1)
model.add(layer2)
model.add(layer3)

model.compile("cross_entropy", 0.5)

costs = model.train(X, Y, 10000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.show()

predictions = model.predict(X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

