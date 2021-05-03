import numpy as np
from DLModel import *

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
