from unit10 import c1w3_utils as u10
from DL1 import *
X, Y = u10.load_planar_dataset()

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = u10.load_extra_datasets()
datasets = {"noisy_circles": noisy_circles, "noisy_moons": noisy_moons, "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "gaussian_quantiles"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y % 2

layer1 = DLLayer("layer 1", 17, (2,), "relu", "random", 0.01, "adaptive")
layer2 = DLLayer("layer 2", 1, (17,), "sigmoid", "random", 0.1)

model = DLModel()
model.add(layer1)
model.add(layer2)

model.compile("cross_entropy", 0.5)

model.train(X, Y, 100)

u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.show()

predictions = model.predict(X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')