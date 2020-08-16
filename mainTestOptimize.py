from NeuralNetwork import NeuralNetwork
from layers.FullyConnectedLayer import FullyConnectedLayer
from utils.activationFunctions import sigmoid
import gzip
import pickle as cPickle
from utils.labelsToMat import labelsToMat

# Loads the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
u = cPickle._Unpickler(f)
u.encoding = 'latin1'
train, val, test = u.load()
f.close()

# Saves the dataset in python lists
X_input = train[0]
Y = labelsToMat(train[1], 10)

# creation of the FC layers
inputDim = 28 * 28
outputDim = 100
actFun = sigmoid
fc1 = FullyConnectedLayer(inputDim, outputDim, actFun)
inputDim = outputDim
outputDim = 75
fc2 = FullyConnectedLayer(inputDim, outputDim, actFun)
inputDim = outputDim
outputDim = 40
fc3 = FullyConnectedLayer(inputDim, outputDim, actFun)
inputDim = outputDim
outputDim = 10
fc4 = FullyConnectedLayer(inputDim, outputDim, actFun)

# creation of the NN
NN = NeuralNetwork([fc1, fc2, fc3, fc4])
alpha = 0.6
NN.optimize(Y, X_input, trainString="inertia", iterMax=5000, alpha=alpha, batchSize=256, costString="MaxL",
            nbElts=50000, decayRate=196000)

error = NN.computeError(labelsToMat(test[1], 10), test[0])

print("the success rate is worth " + str(error) + "%")
