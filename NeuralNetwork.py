import numpy as np
import copy


class NeuralNetwork:

    def __init__(self, layerList, regParam=0.001, regString="L2"):

        self.layerList = layerList
        self.regParam = regParam
        self.regString = regString
        self.nbLayer = len(layerList)
        nbParams = 0
        for layerIndex in range(self.nbLayer):
            if self.layerList[layerIndex].hasWeights == True:
                nbParams += np.prod(self.layerList[layerIndex].weights.shape)

        self.nbParams = nbParams

    def forwardProp(self, X_input, nbElts=1000):
        n_ex = X_input.shape[0]
        FinalActivation = []

        if n_ex % nbElts == 0:
            nbIter = n_ex // nbElts
        else:
            nbIter = n_ex // nbElts + 1

        for iterIndex in range(nbIter):
            activation = X_input[iterIndex * nbElts:(iterIndex + 1) * nbElts]

            for layerIndex in range(self.nbLayer):
                layer = self.layerList[layerIndex]
                activation = layer.forward(activation)

            FinalActivation.append(activation)

        self.activation = np.concatenate(FinalActivation)
        return self.activation

    def backwardProp(self, Y):

        grad = self.activation - Y

        for layerIndex in range(self.nbLayer - 1, -1, -1):
            layer = self.layerList[layerIndex]
            grad = layer.backward(grad, regParam=self.regParam, regString=self.regString)

    # flattens parameters that come under the form of a list from which the element at position layerIndex
    # have the same shape as the weights of the layer n° layerIndex (useful to change the weights like in gradient descent)
    def flattenParams(self, paramsToFlatten):
        paramList = []

        for layerIndex in range(self.nbLayer):
            if self.layerList[layerIndex].hasWeights == True:
                paramList.append(paramsToFlatten[layerIndex].flatten())

        return np.array(paramList)

    # returns a flattened version of all the weights (useful to change the weights like in gradient descent)
    def selfFlattenParam(self):

        paramList = []

        for layerIndex in range(self.nbLayer):
            if self.layerList[layerIndex].hasWeights == True:
                paramList.append(self.layerList[layerIndex].weights.flatten())

        return np.concatenate(paramList)

    # flattens the gradients of all the layers, concatenates them and returns the result
    def selfFlattenGrad(self):

        gradList = []

        for layerIndex in range(self.nbLayer):
            if self.layerList[layerIndex].hasWeights == True:
                gradList.append(self.layerList[layerIndex].gradw.flatten())

        return np.concatenate(gradList)

    # sets new weights as they come under the form of a long array (useful to change the weights like in gradient descent)
    def unflattenParam(self, params):

        nbUnflattenParams = 0

        for layerIndex in range(self.nbLayer):
            if self.layerList[layerIndex].hasWeights == True:
                currentShape = self.layerList[layerIndex].weights.shape
                nbWeights = np.prod(currentShape)
                self.layerList[layerIndex].weights = params[nbUnflattenParams:(nbUnflattenParams + nbWeights)].reshape(
                    currentShape)
                nbUnflattenParams += nbWeights

    def cost(self, Y, costString="RMS", nbElts=3000):
        activation = self.activation
        n_ex = Y.shape[0]

        if n_ex % nbElts == 0:
            nbComp = n_ex // nbElts
        else:
            nbComp = n_ex // nbElts + 1

        cost = 0
        for nIter in range(nbComp):
            Yiter = Y[nIter * nbElts:(nIter + 1) * nbElts]
            activationIter = activation[nIter * nbElts:(nIter + 1) * nbElts]
            if costString == "RMS":
                error = Yiter - activationIter
                cost += (1 / n_ex) * 0.5 * np.multiply(error, error).sum()
            if costString == "MaxL":
                cost -= (1 / n_ex) * (np.multiply(Yiter, np.log(activationIter)) + np.multiply(1 - Yiter, np.log(
                    1 - activationIter))).sum()
        return cost + (0.5 * self.regParam / n_ex) * np.linalg.norm(self.selfFlattenParam()) ** 2

    def costNprop(self, Y, X_input, costString="RMS", nbElts=3000):
        self.forwardProp(X_input, nbElts=nbElts)
        return self.cost(Y, costString, nbElts=nbElts)

    def numGrad(self, Y, X_input, epsilon, costString="RMS"):

        numGrad = np.zeros(self.nbParams)
        costPlusEpsilon = 0.
        costMinusEpsilon = 0.
        originalWeight = 0
        for paramIndex in range(self.nbParams):
            weightList = copy.deepcopy(self.selfFlattenParam())
            originalWeight = weightList[paramIndex]

            weightList[paramIndex] = originalWeight + epsilon
            self.unflattenParam(weightList)
            costPlusEpsilon = self.costNprop(Y, X_input, costString="RMS")

            weightList[paramIndex] = originalWeight - epsilon
            self.unflattenParam(weightList)
            costMinusEpsilon = self.costNprop(Y, X_input, costString="RMS")

            numGrad[paramIndex] = 1 / (2 * epsilon) * (costPlusEpsilon - costMinusEpsilon)

            weightList[paramIndex] = originalWeight
            self.unflattenParam(weightList)

        return numGrad

    def optimize(self, Y, X_input, trainString="inertia", alpha=0.01, beta=0.9, gamma=0.9, batchSize=128, iterMax=100,
                 costString="RMS", decayRate=10000000, nbElts=1000):
        n_ex = Y.shape[0]
        idx = np.array(range(n_ex))
        nonZero = 1
        if n_ex % batchSize == 0:
            nonZero = 0

        nbBatch = n_ex // batchSize + nonZero

        for iterIndex in range(iterMax):
            Y = Y[idx]
            X_input = X_input[idx]

            grad = self.selfFlattenGrad()

            if trainString == "inertia":
                inertiaGrad = np.zeros(grad.shape, dtype=float)
            elif trainString == "RMSprop":
                RMSvect = np.zeros(grad.shape, dtype=float)
            elif trainString == "adam":
                RMSvect = np.zeros(grad.shape, dtype=float)
                update = np.zeros(grad.shape, dtype=float)

            print("algorithm " + trainString + ": iteration n°" + str(iterIndex) + "| cost = " + str(
                self.costNprop(Y, X_input, costString=costString, nbElts=nbElts)))
            for batchIndex in range(nbBatch):

                Y_LabelsBatch = Y[batchIndex * batchSize:(batchIndex * batchSize + batchSize)]
                X_InputsBatch = X_input[batchIndex * batchSize:(batchIndex * batchSize + batchSize)]
                # computes batch gradient
                self.forwardProp(X_InputsBatch)
                self.backwardProp(Y_LabelsBatch)
                grad = self.selfFlattenGrad()
                mult = 1 / (1 + (iterIndex * batchSize + batchIndex) / decayRate)
                if trainString == "batchGrad":
                    self.unflattenParam(self.selfFlattenParam() - mult * alpha * grad)

                if trainString == "inertia":
                    inertiaGrad = beta * inertiaGrad + (1 - beta) * grad
                    self.unflattenParam(self.selfFlattenParam() - mult * alpha * inertiaGrad)

                if trainString == "RSMprop":
                    RMSvect = gamma * RMSvect + (1 - gamma) * np.multiply(grad, grad)
                    self.unflattenParam(
                        self.selfFlattenParam() - mult * alpha * np.divide(grad, (RMSvect + 10 ** (-8))))

                if trainString == "adam":
                    RMSvect = gamma * RMSvect + (1 - gamma) * np.multiply(grad, grad)
                    update = beta * update + (1 - beta) * grad
                    self.unflattenParam(
                        self.selfFlattenParam() - mult * alpha * np.divide(update / (1 - beta ** (batchIndex + 1)), (
                                RMSvect / (1 - gamma ** (batchIndex + 1)) + 10 ** (-8))))

            np.random.shuffle(idx)

    def computeError(self, Y_Labels, X_Inputs):

        n_ex = np.shape(Y_Labels)[0]

        self.forwardProp(X_Inputs)

        outputs = self.activation

        outputs = (outputs > 0.5)
        outputs = outputs.astype(int)

        errorVect = np.abs(outputs - Y_Labels).sum(axis=1)
        errorVect = errorVect > 0
        errorVect = errorVect.astype(int)
        nbError = np.sum(errorVect)

        return (1 - nbError / n_ex) * 100
