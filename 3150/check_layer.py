from DL1 import *
import numpy as np
import random

# -------------------- Exercise 1.1 --------------------
np.random.seed(1)

l = [None]

l.append(DLLayer("Hidden 1", 6, (4000,)))
print(l[1])

l.append(DLLayer("Hidden 2", 12, (6,), "leaky_relu", "random", 0.5, "adaptive"))

l[2].adaptive_cont = 1.2
print(l[2])

l.append(DLLayer("Neurons 3", 16, (12,), "tanh"))
print(l[3])

l.append(DLLayer("Neurons 4", 3, (16,), "sigmoid", "random", 0.2, "adaptive"))
l[4].random_scale = 10.0
l[4].init_weights("random")
print(l[4])


# -------------------- Exercise 1.2 --------------------
Z = np.array([[1, -2, 3, -4], [-10, 20, 30, -40]])

l[2].leaky_relu_d = 0.1

for i in range(1, len(l)):
    print(l[i].activation_forward(Z))


# -------------------- Exercise 1.3 --------------------
np.random.seed(2)
m = 3
X = np.random.randn(4000, m)
Al = X

for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    print('layer', i, " A", str(Al.shape), ":\n", Al)

# -------------------- Exercise 1.4 --------------------
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    dZ = l[i].activation_backward(Al)
    print('layer', i, " dZ", str(dZ.shape), ":\n", dZ)

# -------------------- Exercise 1.5 --------------------
Al = X

for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)

np.random.seed(3)
fig, axes = plt.subplots(1, 4, figsize=(12, 16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
dAl = np.random.randn(Al.shape[0], m) * np.random.randint(-100, 100 + 1, Al.shape)

for i in reversed(range(1, len(l))):
    axes[i - 1].hist(dAl.reshape(-1), align='left')
    axes[i - 1].set_title('dAl[' + str(i) + ']')
    dAl = l[i].backward_propagation(dAl)

plt.show()

# -------------------- Exercise 1.6 --------------------
np.random.seed(4)
random.seed(4)
l1 = DLLayer("Hidden1", 3, (4,), "trim_sigmoid", "zeros", 0.2, "adaptive")
l2 = DLLayer("Hidden2", 2, (3,), "relu", "random", 1.5)

print("before update W1:\n", str(l1.W), "\nb1.T:\n", str(l1.b.T))
print("W2:\n", str(l2.W), "\nb2.T:\n", str(l2.b.T))

l1.dW = np.random.randn(3, 4) * random.randrange(-100, 100)
l1.db = np.random.randn(3, 1) * random.randrange(-100, 100)
l2.dW = np.random.randn(2, 3) * random.randrange(-100, 100)
l2.db = np.random.randn(2, 1) * random.randrange(-100, 100)

l1.update_parameters()
l2.update_parameters()

print("after update W1:\n", str(l1.W), "\nb1.T:\n", str(l1.b.T))
print("W2:\n", str(l2.W), "\nb2.T:\n", str(l2.b.T))