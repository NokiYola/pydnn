import numpy as np


class MaxPool:
    hasWeights = False

    def __init__(self, size1, size2):
        self.size = (size1, size2)

    def forward(self, X_input):
        n_ex = X_input.shape[0]
        channels = X_input.shape[1]
        dimg1 = X_input.shape[2]
        dimg2 = X_input.shape[3]
        size1, size2 = self.size
        self.inputShape = (n_ex, channels, dimg1, dimg2)
        firstWindow = np.array([i * dimg2 + j for i in range(size1) for j in range(size2)])

        upperLeftCorners = size1 * dimg2 * np.array(range(dimg1 // size1))[:, None] + size2 * np.array(
            range(dimg2 // size2))[None, :]

        indexOneChannel = upperLeftCorners[:, :, None] + firstWindow[None, None, :]
        channelDim = np.array([dimg1 * dimg2 * c for c in range(channels)])

        indexOneExample = channelDim[:, None, None, None] + indexOneChannel[None, :, :, :]
        exampleDim = np.array([dimg1 * dimg2 * channels * m for m in range(n_ex)])

        poolIndexes = exampleDim[:, None, None, None, None] + indexOneExample[None, :, :, :, :]

        poolMat = X_input.take(poolIndexes)

        self.maxIndexes = np.argmax(poolMat, axis=(len(poolMat.shape) - 1))

        return np.max(poolMat, axis=(len(poolMat.shape) - 1))

    def backward(self, gradY, regParam=0.1, regString="L2"):
        MaxIdx = self.maxIndexes
        if gradY.shape != MaxIdx.shape:
            gradY = gradY.reshape(MaxIdx.shape)

        n_ex, channels, dimg1, dimg2 = self.inputShape
        size1, size2 = self.size
        dimOut2 = gradY.shape[3]
        dimOut1 = gradY.shape[2]

        gradX = np.zeros(np.prod(self.inputShape), dtype=float)

        idxFlat = dimg2 * (np.array(range(dimOut1)) * size1)[:, None] + size2 * np.array(range(dimOut2))[None, :]
        idxFlat = idxFlat[None, :, :] + dimg1 * dimg2 * np.array(range(channels))[:, None, None]
        idxFlat = idxFlat[None, :, :, :] + channels * dimg1 * dimg2 * np.array(range(n_ex))[:, None, None, None]
        idxFlat += np.floor_divide(MaxIdx, size2) * dimg2 + np.divmod(MaxIdx, size2)[1]
        idxFlat = idxFlat.flatten()
        gradX[idxFlat] = gradY.flatten()
        return gradX.reshape(self.inputShape)
