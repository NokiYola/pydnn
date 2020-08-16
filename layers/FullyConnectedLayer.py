import numpy as np


class FullyConnectedLayer:
    hasWeights = True

    def __init__(self, inputDim, outputDim, actFun):

        self.inputDim = inputDim
        self.outputDim = outputDim
        self.weights = 2 * (np.random.rand(outputDim, inputDim + 1) - 0.5)
        self.actFun = actFun
        self.gradw = np.zeros(self.weights.shape, dtype=float)

    def forward(self, X_input):
        n_ex = X_input.shape[0]
        if len(X_input.shape) > 2:
            inputSize = X_input.size // n_ex
            self.activation = X_input.reshape(n_ex, inputSize)
        else:
            self.activation = X_input

        z = np.dot(self.weights, np.concatenate((np.ones(n_ex)[None, :], self.activation.T), axis=0)).T
        self.z = z

        return self.actFun.fun(z)

    def backward(self, delta, regParam=0.1, regString="L2"):

        n_ex = len(self.activation)
        delta = np.multiply(self.actFun.der(self.z), delta).T
        if regString == "L2":
            self.gradw = (1 / n_ex) * np.dot(delta,
                                             np.concatenate((np.ones(n_ex)[:, None], self.activation), axis=1)) + (
                                 regParam / n_ex) * self.weights

        delta = np.dot(np.delete(self.weights, (0), axis=1).T, delta)

        return delta.T
