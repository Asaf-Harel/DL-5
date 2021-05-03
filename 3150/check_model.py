import numpy as np
from DL1 import *

# -------------------- Exercise 2.1 - 2.3 --------------------
np.random.seed(1)

m1 = DLModel()
AL = np.random.rand(4, 3)
Y = np.random.rand(4, 3) > 0.7
m1.compile("cross_entropy")
errors = m1.loss_forward(AL, Y)
dAL = m1.loss_backward(AL, Y)
print("cross entropy error:\n", errors)
print("cross entropy dAL:\n", dAL)

m2 = DLModel()
m2.compile("squared_means")
errors = m2.loss_forward(AL, Y)
dAL = m2.loss_backward(AL, Y)
print("squared means error:\n", errors)
print("squared means dAL:\n", dAL)

# -------------------- Exercise 2.4 --------------------
print("cost m1:", m1.compute_cost(AL, Y))
print("cost m2:", m2.compute_cost(AL, Y))

# -------------------- Exercise 2.5 --------------------
np.random.seed(1)
model = DLModel()
model.add(DLLayer("Perseptrons 1", 10, (12288,)))
model.add(DLLayer("Perseptrons 2", 1, (10,), "trim_sigmoid"))
model.compile("cross_entropy", 0.7)
X = np.random.randn(12288, 10) * 256
print("predict:", model.predict(X))

# -------------------- Exercise 2.6 --------------------
print(model)
