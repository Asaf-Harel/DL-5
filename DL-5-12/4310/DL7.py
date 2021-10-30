import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix


class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=0.01,
                 optimization=None, regularization=None):
        # Constant parameters
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = '_'.join(activation.lower().split(' '))
        self._learning_rate = learning_rate
        self._optimization = optimization.lower() if optimization else optimization
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

        elif self._optimization == "adam":
            self._adam_v_dW = np.zeros(self.W.shape)
            self._adam_v_db = np.zeros(self.b.shape)

            self._adam_s_dW = np.zeros(self.W.shape)
            self._adam_s_db = np.zeros(self.b.shape)

            self.adam_beta1 = 0.9
            self.adam_beta2 = 0.999
            self.adam_epsilon = np.exp(-8)

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
            self.L2_lambda = 0.6
        elif self.regularization == 'dropout':
            self.dropout_keep_prob = 0.6

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units, 1), dtype=float)

        if W_initialization.lower() == "zeros":
            self.W = np.full(*self._get_W_shape(), self.alpha)
        elif W_initialization.lower() == "random":
            self.W = np.random.randn(*self._get_W_shape()) * self.random_scale
        elif W_initialization.lower() == "xavier":
            self.W = np.random.randn(*self._get_W_shape()) * np.sqrt(1 / self._get_W_init_factor())
        elif W_initialization.lower() == "he":
            self.W = np.random.randn(*self._get_W_shape()) * np.sqrt(2 / self._get_W_init_factor())
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except FileNotFoundError:
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    def _get_W_init_factor(self):
        return np.sum(self._input_shape)

    def _get_W_shape(self):
        return self._num_units, *self._input_shape

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

    @staticmethod
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

    @staticmethod
    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]

        eps = np.exp(-10)
        m = np.where(m == 0, eps, m)  # to avoid divide by zero

        self.dW = (1.0 / m) * np.dot(dZ, self._A_prev.T)
        self.dW = self.backward_l2(dZ)

        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

        dA_Prev = np.dot(self.W.T, dZ)
        dA_Prev = self.backward_dropout(dA_Prev)

        return dA_Prev

    def update_parameters(self, t):
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

        elif self._optimization == "adam":
            self._adam_v_dW = self.adam_beta1 * self._adam_v_dW + ((1 - self.adam_beta1) * self.dW)
            adam_v_dW_A = self._adam_v_dW / (1 - (self.adam_beta1 ** t))

            self._adam_v_db = self.adam_beta1 * self._adam_v_db + ((1 - self.adam_beta1) * self.db)
            adam_v_db_A = self._adam_v_db / (1 - (self.adam_beta1 ** t))

            self._adam_s_dW = self.adam_beta2 * self._adam_s_dW + (1 - self.adam_beta2) * (self.dW ** 2)
            adam_s_dW_A = self._adam_s_dW / (1 - (self.adam_beta2 ** t))

            self._adam_s_db = self.adam_beta2 * self._adam_s_db + (1 - self.adam_beta2) * (self.db ** 2)
            adam_s_db_A = self._adam_s_db / (1 - (self.adam_beta2 ** t))

            self.W -= (self.alpha * adam_v_dW_A) / (np.sqrt(adam_s_dW_A + self.adam_epsilon))
            self.b -= (self.alpha * adam_v_db_A) / (np.sqrt(adam_s_db_A + self.adam_epsilon))

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

        if self._activation != 'noactivation':
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
        elif self._optimization == "adam":
            s += "\t\tadam parameters:\n"
            s += "\t\t\tbeta1: " + str(self.adam_beta1) + "\n"
            s += "\t\t\tbeta2: " + str(self.adam_beta2) + "\n"
            s += "\t\t\tepsilon: " + str(self.adam_epsilon) + "\n"

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
        s += "\tparameters:\n\t\tweights shape: " + str(self.W.shape) + "\n"
        s += "\t\tb shape: " + str(self.b.shape) + "\n"

        return s


class DLConv(DLLayer):
    def __init__(self, name, num_filters, input_shape, filter_size, strides, padding, activation="relu",
                 W_initialization="He", learning_rate=0.01, optimization="adam", regularization=None):
        self.num_filters = num_filters
        self._input_shape = input_shape
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding if not isinstance(padding, str) else padding.lower()

        if self.padding == 'same':
            padding_h = (strides[0] * self._input_shape[1] - strides[0] - self._input_shape[1] + self.filter_size[
                0] + 1) // 2
            padding_w = (strides[1] * self._input_shape[2] - strides[1] - self._input_shape[2] + self.filter_size[
                0] + 1) // 2
            self.padding = (padding_w, padding_h)
        elif self.padding == 'valid':
            self.padding = (0, 0)
        else:
            self.padding = padding

        self.h_out = int(((self._input_shape[1] + 2 * self.padding[0] - filter_size[0]) // strides[0]) + 1)
        self.w_out = int(((self._input_shape[2] + 2 * self.padding[1] - filter_size[1]) // strides[1]) + 1)

        super().__init__(name, num_filters, input_shape, activation, W_initialization, learning_rate,
                         optimization, regularization)

    def _get_W_shape(self):
        return self._num_units, self._input_shape[0], self.filter_size[0], self.filter_size[1]

    def _get_W_init_factor(self):
        return self._input_shape[0] * self.filter_size[0] * self.filter_size[1]

    @staticmethod
    def im2col_indices(A, filter_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        A_padded = np.pad(A, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[0], padding[1])), mode='constant',
                          constant_values=(0, 0))

        k, i, j = DLConv.get_im2col_indices(A.shape, filter_size, padding, stride)

        cols = A_padded[:, k, i, j]
        C = A.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(filter_size[0] * filter_size[1] * C, -1)
        return cols

    @staticmethod
    def get_im2col_indices(A_shape, filter_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        # First figure out what the size of the output should be
        m, C, H, W = A_shape
        out_height = int((H + 2 * padding[0] - filter_size[0]) / stride[0]) + 1
        out_width = int((W + 2 * padding[1] - filter_size[1]) / stride[1]) + 1

        i0 = np.repeat(np.arange(filter_size[0]), filter_size[1])
        i0 = np.tile(i0, C)
        i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(filter_size[1]), filter_size[0] * C)
        j1 = stride[1] * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), filter_size[0] * filter_size[1]).reshape(-1, 1)

        return k, i, j

    @staticmethod
    def col2im_indices(cols, A_shape, filter_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        m, C, H, W = A_shape
        H_padded, W_padded = H + 2 * padding[0], W + 2 * padding[1]
        A_padded = np.zeros((m, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = DLConv.get_im2col_indices(A_shape, filter_size, padding, stride)

        cols_reshaped = cols.reshape(C * filter_size[0] * filter_size[1], -1, m)

        cols_reshaped = cols_reshaped.transpose(2, 0, 1)

        np.add.at(A_padded, (slice(None), k, i, j), cols_reshaped)

        if padding[0] == 0 and padding[1] == 0:
            return A_padded
        if padding[0] == 0:
            return A_padded[:, :, :, padding[1]:-padding[1]]
        if padding[1] == 0:
            return A_padded[:, :, padding[0]:-padding[0], :]
        return A_padded[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    @staticmethod
    def forward_propagation(self, A_prev):
        A_prev = np.transpose(A_prev, (3, 0, 1, 2))
        A_prev = self.im2col_indices(A_prev, self.filter_size, self.padding, self.strides)

        temp_W = self.W
        self.W = self.W.reshape(self.num_filters, -1)

        A = super().forward_propagation(self, A_prev)
        A = A.reshape(self.num_filters, self.h_out, self.w_out, -1)

        self.W = temp_W
        return A

    @staticmethod
    def backward_propagation(self, dA):
        m = dA.shape[-1]
        W_temp = self.W

        dA = dA.reshape(self.num_filters, -1)
        self.W = self.W.reshape(self.num_filters, -1)

        dA_prev = super().backward_propagation(self, dA)

        self.W = W_temp
        self.dW = self.dW.reshape(self.W.shape)

        prev_A_shape = (m, *self._input_shape)
        dA_prev = self.col2im_indices(dA_prev, prev_A_shape, self.filter_size, self.padding, self.strides)

        # transpose dA-prev from (m,C,H,W) to (C,H,W,m)
        dA_prev = dA_prev.transpose(1, 2, 3, 0)

        return dA_prev

    def __str__(self):
        s = "Convolutional " + super(DLConv, self).__str__()
        s += "\tConvolutional parameters:\n"
        s += f"\t\tfilter size: {self.filter_size}\n"
        s += f"\t\tstrides: {self.strides}\n"
        s += f"\t\tpadding: {self.padding}\n"
        s += f"\t\toutput shape: {(self.num_filters, self.h_out, self.w_out)}\n"

        return s


class DLMaxPooling:
    def __init__(self, name, input_shape, filter_size, strides):
        self._name = name
        self._input_shape = input_shape
        self._filter_size = filter_size
        self._strides = strides

        self.h_out = int((input_shape[1] - self._filter_size[0]) / self._strides[0]) + 1
        self.w_out = int((input_shape[2] - self._filter_size[1]) / self._strides[1]) + 1

    @staticmethod
    def forward_propagation(self, A_prev):
        # first transpose A_prev from (C,H,W,m) to (m,C,H,W)
        A_prev = A_prev.transpose(3, 0, 1, 2)
        m, C, H, W = A_prev.shape
        prev_A = A_prev.reshape(m * C, 1, H, W)

        self.A_prev = DLConv.im2col_indices(prev_A, self._filter_size, padding=(0, 0), stride=self._strides)
        self.max_indexes = np.argmax(self.A_prev, axis=0)

        Z = self.A_prev[self.max_indexes, range(self.max_indexes.size)]
        Z = Z.reshape(self.h_out, self.w_out, m, C).transpose(3, 0, 1, 2)

        return Z

    @staticmethod
    def backward_propagation(self, dZ):
        dA_prev = np.zeros_like(self.A_prev)

        # transpose dZ from C,h,W,C to H,W,m,c and flatten it
        # Then, insert dZ values to dA_prev in the places of the max indexes
        dZ_flat = dZ.transpose(1, 2, 3, 0).ravel()
        dA_prev[self.max_indexes, range(self.max_indexes.size)] = dZ_flat

        # get the original prev_A structure from col2im
        m = dZ.shape[-1]
        C, H, W = self._input_shape
        shape = (m * C, 1, H, W)
        dA_prev = DLConv.col2im_indices(dA_prev, shape, self._filter_size, padding=(0, 0), stride=self._strides)
        dA_prev = dA_prev.reshape(m, C, H, W).transpose(1, 2, 3, 0)
        return dA_prev

    def update_parameters(self):
        return

    def __str__(self):
        s = f"Maxpooling {self._name} Layer:\n"
        s += f"\tinput_shape: {self._input_shape}\n"
        s += "\tMaxpooling parameters:\n"
        s += f"\t\tfilter size: {self._filter_size}\n"
        s += f"\t\tstrides: {self._strides}\n"

        # number of output channels == number of input channels
        s += f"\t\toutput shape: {(self._input_shape[0], self.h_out, self.w_out)}\n"

        return s


class DLFlatten:
    def __init__(self, name, input_shape):
        self._name = name
        self._input_shape = input_shape

    @staticmethod
    def forward_propagation(self, prev_A):
        m = prev_A.shape[-1]
        A = np.copy(prev_A.reshape(-1, m))

        return A

    @staticmethod
    def backward_propagation(self, dA):
        m = dA.shape[-1]
        dA_prev = np.copy(dA.reshape(*self._input_shape, m))

        return dA_prev

    def update_parameters(self):
        return

    def __str__(self):
        s = f"Flatten {self._name} Layer:\n"
        s += f"\tinput_shape: {self._input_shape}\n"

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
        eps = np.exp(-10)
        AL = np.where(AL == 0, eps, AL)
        AL = np.where(AL == 1, 1 - eps, AL)

        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return error

    def _cross_entropy_backward(self, AL, Y):
        eps = np.exp(-10)
        AL = np.where(AL == 0, eps, AL)
        AL = np.where(AL == 1, 1 - eps, AL)

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
        self.loss = '_'.join(loss.lower().split(' '))

        if loss == "squared_means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif loss == "categorical_cross_entropy":
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
        elif loss == "cross_entropy":
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
            self.layers[l].update_parameters(self._t)
        return dAl_t

    def set_train(self, is_train):
        self.is_train = is_train

        for i in range(1, len(self.layers)):
            self.layers[i].set_train(is_train)

    def train(self, X, Y, num_epochs, mini_batch_size):
        self.set_train(True)
        print_ind = max(num_epochs // 100, 1)
        costs = []
        seed = 10

        for i in range(num_epochs):
            self._t = 1

            Al = np.array(X, copy=True)

            mini_batches = self.random_mini_batches(X, Y, mini_batch_size, seed)
            seed += 1

            for mini_batch in mini_batches:
                if len(mini_batch[0][0]) == 0:
                    continue

                Al = self.forward_propagation(mini_batch[0])
                dAl = self.backward_propagation(Al, mini_batch[1])

                # record progress
                if num_epochs == 1 or (i > 0 and i % print_ind == 0):
                    J = self.compute_cost(Al, mini_batch[1])
                    costs.append(J)
                    inject_string = ""
                    if self.inject_str_func is not None:
                        inject_string = self.inject_str_func(self, X, Y, Al)
                    print(f"cost after {i} full updates {100 * i / num_epochs}%:{J}" + inject_string)

        self.set_train(False)
        return costs

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        np.random.seed(seed)

        m = X.shape[1]

        permutation = list(np.random.permutation(m))

        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((-1, m))
        num_complete_minibatches = m // mini_batch_size

        mini_batches = []

        for k in range(num_complete_minibatches + 1):
            mini_batch_X = shuffled_X[:, mini_batch_size * k: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, mini_batch_size * k: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
            if (k == num_complete_minibatches + 1) and ((m / mini_batch_size) != num_complete_minibatches):
                mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size:]
                mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size:]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

        return mini_batches

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1, L):
            Al = self.layers[i].forward_propagation(Al, True)

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
