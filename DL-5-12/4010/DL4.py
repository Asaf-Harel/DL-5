import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix


class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2,
                 optimization=None, regularization=None):
        # Constant parameters
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation.lower()
        self._learning_rate = learning_rate
        self._optimization = optimization.lower if optimization else optimization
        self.alpha = learning_rate
        self.random_scale = 0.01
        self.is_train = False
        self.regularization = regularization.lower() if regularization else optimization
        self.init_weights(W_initialization)

        # Optimization parameters
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, *self._input_shape), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = -0.5

        # Activation
        self.activation_trim = 1e-10

        if self._activation == "sigmoid":
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward

        elif self._activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward

        elif self._activation == "tanh":
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward

        elif self._activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward

        elif self._activation == "relu":
            self.activation_forward = self._relu
            self.activation_backward = self._relu_backward

        elif self._activation == "leaky_relu":
            self.leaky_relu_d = 0.01
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward

        elif self._activation == "softmax":
            self.activation_forward = self._softmax
            self.activation_backward = self._softmax_backward

        elif self._activation == "trim_softmax":
            self.activation_forward = self._trim_softmax
            self.activation_backward = self._softmax_backward

        elif self._activation == "no_activation":
            self.activation_forward = self._no_activation
            self.activation_backward = self._no_activation_backward

        # Regularization parameters
        if self.regularization == 'l2':
            self.L2_lambda = 0.6  # TODO: change back to 0.1
        elif self.regularization == 'dropout':
            self.dropout_keep_prob = 0.6

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units, 1), dtype=float)

        if W_initialization.lower() == "zeros":
            self.W = np.full((self._num_units, *self._input_shape), self.alpha)
        elif W_initialization.lower() == "random":
            self.W = np.random.randn(*(self._num_units, *self._input_shape)) * self.random_scale
        elif W_initialization.lower() == "xavier":
            prev_l = self._input_shape[0]
            self.W = np.random.randn(self._num_units, prev_l) * np.sqrt(1 / prev_l)
        elif W_initialization.lower() == "he":
            prev_l = self._input_shape[0]
            self.W = np.random.randn(self._num_units, prev_l) * np.sqrt(2 / prev_l)
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except FileNotFoundError:
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        return dA * A * (1 - A)

    def _trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1 / (1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = A = 1 / (1 + np.exp(-Z))
            TRIM = self.activation_trim
            if TRIM > 0:
                A = np.where(A < TRIM, TRIM, A)
                A = np.where(A > 1 - TRIM, 1 - TRIM, A)
            return A

    def _trim_sigmoid_backward(self, dA):
        A = self._trim_sigmoid(self._Z)

        return dA * A * (1 - A)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        return dA * (1 - A ** 2)

    def _trim_tanh(self, Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def _trim_tanh_backward(self, dA):
        A = self._trim_tanh(self._Z)

        return dA * (1 - A ** 2)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_backward(self, dA):
        return np.where(self._Z <= 0, 0, dA)

    def _leaky_relu(self, Z):
        return np.where(Z <= 0, Z * self.leaky_relu_d, Z)

    def _leaky_relu_backward(self, dA):
        return np.where(self._Z <= 0, dA * self.leaky_relu_d, dA)

    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def _softmax_backward(self, dZ):
        return dZ

    def _trim_softmax(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100, Z)
                eZ = np.exp(Z)
        A = eZ / np.sum(eZ, axis=0)
        return A

    def _no_activation(self, Z):
        return Z

    def _no_activation_backward(self, dZ):
        return dZ

    def forward_dropout(self, A_prev):
        if self.is_train and self.regularization == 'dropout':
            self._D = np.random.rand(*A_prev.shape)
            self._D = np.where(self._D < self.dropout_keep_prob, 1, 0)
            A_prev *= self._D
            A_prev /= self.dropout_keep_prob
        return np.array(A_prev, copy=True)

    def forward_propagation(self, A_prev):
        self._A_prev = self.forward_dropout(A_prev)
        self._Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_forward(self._Z)

        return A

    def backward_dropout(self, dA_prev):
        if self.regularization == 'dropout':
            dA_prev *= self._D
            dA_prev /= self.dropout_keep_prob
        return dA_prev

    def backward_l2(self, dZ):
        m = dZ.shape[-1]

        if self.regularization == 'l2':
            return self.dW + ((self.L2_lambda * self.W) / m)
        return self.dW

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]

        self.dW = (1.0 / m) * np.dot(dZ, self._A_prev.T)
        self.dW = self.backward_l2(dZ)
        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_Prev = np.dot(self.W.T, dZ)
        dA_Prev = self.backward_dropout(dA_Prev)

        return dA_Prev

    def update_parameters(self):
        if self._optimization is None:
            self.W -= self.dW * self.alpha
            self.b -= self.db * self.alpha
        elif self._optimization == "adaptive":
            self._adaptive_alpha_W *= np.where(self._adaptive_alpha_W * self.dW > 0, self.adaptive_cont,
                                               self.adaptive_switch)
            self._adaptive_alpha_b *= np.where(self._adaptive_alpha_b * self.db > 0, self.adaptive_cont,
                                               self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b

    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(f"{path}/{file_name}.h5", 'w') as hf:
            hf.create_dataset('W', data=self.W)
            hf.create_dataset('b', data=self.b)

    def set_train(self, is_train):
        self.is_train = is_train

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"

        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d) + "\n"

        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"

        # optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont) + "\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch) + "\n"

        # regularization
        if self.regularization == 'l2':
            s += "\tregularization: L2\n"
            s += "\t\tL2 parameters:\n"
            s += "\t\t\tlambda: " + str(self.L2_lambda) + "\n"

        if self.regularization == 'dropout':
            s += "\tregularization: dropout\n"
            s += "\t\tDropout parameters:\n"
            s += "\t\t\tkeep prob: " + str(self.dropout_keep_prob) + "\n"

        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape) + "\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()

        return s


class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False
        self.is_train = False
        self.inject_str_func = None

    def add(self, layer):
        self.layers.append(layer)

    def _squared_means(self, AL, Y):
        return (AL - Y) ** 2

    def _squared_means_backward(self, AL, Y):
        return 2 * (AL - Y)

    def _cross_entropy(self, AL, Y):
        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return error

    def _cross_entropy_backward(self, AL, Y):
        dAL = np.where(Y == 0, 1 / (1 - AL), -1 / AL)
        return dAL

    def _categorical_cross_entropy(self, AL, Y):
        errors = np.where(Y == 1, -np.log(AL), 0)
        return errors

    def _categorical_cross_entropy_backward(self, AL, Y):
        dA = AL - Y
        return dA

    def compile(self, loss, threshold=0.5):
        self.threshold = threshold
        self.loss = loss.lower()

        if "squared" in loss and "means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif "categorical" in loss and "cross" in loss and "entropy" in loss:
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
        elif "cross" in loss and "entropy" in loss:
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward

        self._is_compiled = True

    def regularization_cost(self, m):
        costs = 0
        L2_lambda = 0

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            if layer.regularization == 'l2':
                L2_lambda = layer.L2_lambda
                costs += np.sum(np.square(layer.W))
        return (L2_lambda * costs) / (2 * m)

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)

        return (np.sum(errors) / m) + self.regularization_cost(m)

    def forward_propagation(self, X):
        L = len(self.layers)
        for l in range(1, L):
            X = self.layers[l].forward_propagation(X)
        return X

    def backward_propagation(self, Al, Y):
        L = len(self.layers)
        dAl_t = self.loss_backward(Al, Y)

        for l in reversed(range(1, L)):
            dAl_t = self.layers[l].backward_propagation(dAl_t)
            self.layers[l].update_parameters()
        return dAl_t

    def set_train(self, is_train):
        self.is_train = is_train

        for i in range(1, len(self.layers)):
            self.layers[i].set_train(is_train)

    def train(self, X, Y, num_iterations):
        self.set_train(True)
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []

        for i in range(num_iterations):
            Al = np.array(X, copy=True)
            Al = self.forward_propagation(Al)
            dAl = self.backward_propagation(Al, Y)

            # record progress
            if num_iterations == 1 or (i > 0 and i % print_ind == 0):
                J = self.compute_cost(Al, Y)
                costs.append(J)
                inject_string = ""
                if self.inject_str_func is not None:
                    inject_string = self.inject_str_func(self, X, Y, Al)
                print(f"cost after {i} full updates {100 * i / num_iterations}%:{J}" + inject_string)
        return costs

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1, L):
            Al = self.layers[i].forward_propagation(Al)

        if Al.shape[0] > 1:
            return np.where(Al == Al.max(axis=0), 1, 0)
        return Al > self.threshold

    def confusion_matrix(self, X, Y):
        AL = self.predict(X)
        predictions = np.argmax(AL, axis=0)
        labels = np.argmax(Y, axis=0)

        return confusion_matrix(predictions, labels)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(1, len(self.layers)):
            self.layers[i].save_weights(path, f"Layer{i}")

    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers) - 1) + "\n"

        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) + "\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1, len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"

        return s
