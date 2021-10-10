import numpy as np
import matplotlib.pyplot as plt


class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2,
                 optimization=None):
        # Constant parameters
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._learning_rate = learning_rate
        self._optimization = optimization
        self.alpha = learning_rate
        self.random_scale = 0.01

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

        self.init_weights(W_initialization)

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units, 1), dtype=float)

        if W_initialization == "zeros":
            self.W = np.full((self._num_units, *self._input_shape), self.alpha)
        elif W_initialization == "random":
            self.W = np.random.randn(*(self._num_units, *self._input_shape)) * self.random_scale

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

    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_forward(self._Z)

        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]
        self.dW = (1.0 / m) * np.dot(dZ, self._A_prev.T)
        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_Prev = np.dot(self.W.T, dZ)

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

    def compile(self, loss, threshold=0.5):
        self.threshold = threshold
        self.loss = loss

        if loss == "squared_means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif loss == "cross_entropy":
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward

        self._is_compiled = True

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)

        return np.sum(errors) / m

    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []

        for i in range(num_iterations):
            # forward propagation
            Al = X
            for l in range(1, L):
                Al = self.layers[l].forward_propagation(Al, False)

            # backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1, L)):
                dAl = self.layers[l].backward_propagation(dAl)

                # update parameters
                self.layers[l].update_parameters()

            # record progress
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
        print("cost after ", str(i // print_ind), "%:", str(J))

        return costs

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1, L):
            Al = self.layers[i].forward_propagation(Al, True)
        return Al > self.threshold

    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers) - 1) + "\n"

        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) + "\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1, len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"

        return s
