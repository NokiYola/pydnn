import numpy as np


# tribute to sylvain Gugger and fast.ai from who some of the most important commands in this code come from
class ConvolutionLayer:
    hasWeights = True

    def __init__(self, actFun, dFilter1, dFilter2, nbFilters, dImg1, dImg2, channels, stride1, stride2, pad1, pad2):

        self.filterDimensions = (channels, dFilter1, dFilter2)
        self.filterStrides = (stride1, stride2)
        self.outputDimension = (nbFilters, int(np.floor((dImg1 + 2 * pad1 - dFilter1) / stride1) + 1),
                                int(np.floor((dImg2 + 2 * pad2 - dFilter2) / stride2) + 1))
        self.inputDimension = (dImg1, dImg2, channels)

        self.weights = 2 * (np.random.rand(channels * dFilter1 * dFilter2 + 1, nbFilters) - 0.5)

        self.padding = (pad1, pad2)
        self.actFun = actFun

        self.gradw = np.zeros(self.weights.shape, dtype=float)

    def arr2vec(self, input):

        channels = self.filterDimensions[0]

        dFilter1 = self.filterDimensions[1]
        dFilter2 = self.filterDimensions[2]

        dImg1 = self.inputDimension[0]
        dImg2 = self.inputDimension[1]

        stride1 = self.filterStrides[0]
        stride2 = self.filterStrides[1]

        pad1 = self.padding[0]
        pad2 = self.padding[1]

        n_ex = input.shape[0]

        if pad1 > 0 or pad2 > 0:
            input_padded = np.zeros((n_ex, channels, dImg1 + 2 * pad1, dImg2 + 2 * pad2))
            input_padded[:, :, pad1:(dImg1 + pad1), pad2:(dImg2 + pad2)] = input.reshape((n_ex, channels, dImg1, dImg2))

        else:
            input_padded = input

        firstConv = (np.array(range(dFilter2))[None, None, :] + (dImg2 + 2 * pad2) * np.array(range(dFilter1))[None, :,
                                                                                     None] + (dImg1 + 2 * pad1) * (
                             dImg2 + 2 * pad2) * np.array(range(channels))[:, None, None]).flatten()

        upperLeftCorner = ((dImg2 + 2 * pad2) * np.array(range(0, dImg1 - dFilter1 + 1 + 2 * pad1, stride1))[:,
                                                None] + np.array(
            range(0, dImg2 - dFilter2 + 1 + 2 * pad2, stride2))).flatten()

        To_Take = upperLeftCorner[:, None] + firstConv[None, :]

        batch = channels * (dImg1 + 2 * pad1) * (dImg2 + 2 * pad2) * np.array(range(n_ex))

        return input_padded.take(batch[:, None, None] + To_Take[None, :, :])

    def vec2arr(self, vec):

        ch = self.filterDimensions[0]

        k1 = self.filterDimensions[1]
        k2 = self.filterDimensions[2]

        dimg1 = self.inputDimension[0]
        dimg2 = self.inputDimension[1]

        o1 = self.outputDimension[1]
        o2 = self.outputDimension[2]

        stride1 = self.filterStrides[0]
        stride2 = self.filterStrides[1]

        pad1 = self.padding[0]
        pad2 = self.padding[1]

        n_ex = vec.shape[0]

        idxx = np.array(range(dimg1))[:, None] - np.array(range(k1))[None, :]
        idxx2 = np.array(range(dimg2))[:, None] - np.array(range(k2))[None, :]

        idxxStack = np.stack([idxx, np.zeros((dimg1, k1), dtype=float)], axis=2)
        idxx2Stack = np.stack([np.zeros((dimg2, k2), dtype=float), idxx2], axis=2)

        idxToReshape = idxxStack[:, None, :, None] + idxx2Stack[None, :, None, :]
        idx = idxToReshape.reshape((dimg1, dimg2, k1 * k2, 2))

        in_bounds = (idx[:, :, :, 0] >= -pad1) * (idx[:, :, :, 0] <= dimg1 - k1 + pad1)
        in_bounds *= (idx[:, :, :, 1] >= -pad2) * (idx[:, :, :, 1] <= dimg2 - k2 + pad2)
        in_strides = (np.mod(idx[:, :, :, 0] + pad1, stride1) == 0) * (np.mod(idx[:, :, :, 1] + pad2, stride2) == 0)
        in_bounds *= in_strides
        in_bounds = (np.ones(ch, dtype=bool)[:, None, None, None] * in_bounds[None, :, :, :])
        in_bounds = np.ones(n_ex, dtype=bool)[:, None, None, None, None] * in_bounds[None, :, :, :, :]

        vecIndexes = np.zeros((n_ex, ch, dimg1, dimg1), dtype=int)
        vecIndexes = (np.floor_divide(idx[:, :, :, 0] + pad1, stride1) * o2 + np.floor_divide(idx[:, :, :, 1] + pad2,
                                                                                              stride2)) * k1 * k2 * ch + (
                             np.array(range(dimg1))[:, None, None] - idx[:, :, :, 0]) * k2 + (
                             np.array(range(dimg2))[None, :, None] - idx[:, :, :, 1])
        vecIndexes = vecIndexes + k1 * k2 * np.array(range(ch))[:, None, None, None]
        vecIndexes = vecIndexes + o1 * o2 * k1 * k2 * ch * np.array(range(n_ex))[:, None, None, None, None]
        vecIndexes = np.divmod(vecIndexes, o1 * o2 * k1 * k2 * ch * n_ex)[1]
        vecIndexes = vecIndexes.astype(int)

        return np.where(in_bounds, np.take(vec, vecIndexes), 0).sum(axis=4)

    def forward(self, input):
        n_ex = input.shape[0]
        vec = self.arr2vec(input)
        self.vec = vec
        weights = np.delete(self.weights, 0, axis=0)
        biases = self.weights[0, :]
        z = (np.matmul(vec, weights) + biases).transpose(0, 2, 1).reshape(
            (n_ex, self.outputDimension[0], self.outputDimension[1], self.outputDimension[2]))
        self.z = z

        return self.actFun.fun(z)

    def backward(self, gradOutput, regParam=0.1, regString="L2"):
        n_ex = gradOutput.shape[0]
        nf, o1, o2 = self.outputDimension

        vec = self.vec

        gradOutput = np.multiply(self.actFun.der(self.z), gradOutput.reshape(self.z.shape))

        grad1 = gradOutput.reshape(n_ex, nf, o1 * o2).transpose(0, 2, 1)

        self.gradw = np.concatenate((grad1.sum(axis=1).mean(axis=0)[None, :],
                                     np.matmul(vec[:, :, :, None], grad1[:, :, None, :]).sum(axis=1).mean(axis=0)),
                                    axis=0) + (regParam / n_ex) * self.weights

        weights = np.delete(self.weights, 0, axis=0)
        return self.vec2arr(np.matmul(grad1, weights.T))
