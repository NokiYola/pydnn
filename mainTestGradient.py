import numpy as np
from NeuralNetwork import NeuralNetwork
from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.ConvolutionLayer import ConvolutionLayer
from utils import createConvLayers
from utils.activationFunctions import sigmoid, id
from layers.MaxPool import MaxPool

# Checking MaxPool grad transmission
print("checking MaxPool grad transmission")
X_input = np.random.rand(4, 3, 32, 22)

mPool = MaxPool(2, 2)
output = mPool.forward(X_input)
depool = mPool.backward(output)

outRes = (X_input == depool).astype(int).mean()
print("ouRes is worth " + str(
    outRes) + " if the img size can be divided by the respective window dimension this should be 0.25 (slightly lower otherwise)")

# ONE FC LAYER GRADIENT CHECK
print("checking the gradient of one fully-connected layer")
nb_ex = 3
inputSize = 7
outputSize = 9
X_input = np.random.rand(nb_ex, inputSize)
FCnumTest = FullyConnectedLayer(inputSize, outputSize, sigmoid)

Y = np.random.rand(nb_ex, outputSize)

NNnumTest = NeuralNetwork([FCnumTest])

epsilon = 0.0001
numGrad = NNnumTest.numGrad(Y, X_input, epsilon)
NNnumTest.forwardProp(X_input)
NNnumTest.backwardProp(Y)
Tgrad = NNnumTest.selfFlattenGrad()

maxError = np.max(np.abs(numGrad - Tgrad))
errorIndex = np.linalg.norm(numGrad - Tgrad) / np.linalg.norm(numGrad + Tgrad)

print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")

# ONE CONVOLUTIONNAL LAYER GRADIENT CHECK
print("checking the gradient of one convolutionnal layer")
# filter dimensions
nbFilters = 15
dFilter1 = 3
dFilter2 = 3

# image dimensions
n_ex = 6
channels = 3
dImg1 = 15
dImg2 = 15

# stride and padding
stride1 = 1
stride2 = 1
pad1 = 2
pad2 = 2

# computation of the output dimensions
outputDim1 = int(np.floor((dImg1 + 2 * pad1 - dFilter1) / stride1) + 1)
outputDim2 = int(np.floor((dImg2 + 2 * pad2 - dFilter2) / stride2) + 1)

# creation of the convolutionnal layer and the NN network
actFun = sigmoid
convNumTest = ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters, dImg1, dImg2, channels, stride1, stride2, pad1,
                               pad2)
NNconvNumtest = NeuralNetwork([convNumTest])

# creation of the input and the output
X_input = np.random.rand(n_ex, channels, dImg1, dImg2)
Y = np.random.rand(n_ex, nbFilters, outputDim1, outputDim2)

# gradient computation
epsilon = 0.0001
numGrad = NNconvNumtest.numGrad(Y, X_input, epsilon)
NNconvNumtest.forwardProp(X_input)
NNconvNumtest.backwardProp(Y)
Tgrad = NNconvNumtest.selfFlattenGrad()

maxError = np.max(np.abs(numGrad - Tgrad))
errorIndex = np.linalg.norm(numGrad - Tgrad) / np.linalg.norm(numGrad + Tgrad)

print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")

# CONVOLUTIONNAL TO CONVOLUTIONNAL GRADIENT CHECK
print("checking the gradient of a convolutionnal to convolutionnal interface")
n_ex = 10
# filter dimensions
nbFilters1 = 5
dFilter1 = 3
dFilter2 = 3
# image dimensions
channels = 3
dimg1 = 15
dimg2 = 15
# creation of the input
X_input = np.random.rand(n_ex, channels, dimg1, dimg2)
# stride and padding
stride1 = 1
stride2 = 1
pad1 = 0
pad2 = 0
# computation of the output dimensions
output1Dim1 = int(np.floor((dimg1 + 2 * pad1 - dFilter1) / stride1) + 1)
output1Dim2 = int(np.floor((dimg2 + 2 * pad2 - dFilter2) / stride2) + 1)
# creation of the first convolutionnal layer
convNum1Test = ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters1, dimg1, dimg2, channels, stride1, stride2, pad1,
                                pad2)

# filter dimensions
nbFilters2 = 4
dFilter1 = 3
dFilter2 = 3
# image dimensions
channels = nbFilters1
dImg1 = output1Dim1
dImg2 = output1Dim2
# stride and padding
stride1 = 1
stride2 = 1
pad1 = 0
pad2 = 0
# computation of the output dimensions
output2Dim1 = int(np.floor((dImg1 + 2 * pad1 - dFilter1) / stride1) + 1)
output2Dim2 = int(np.floor((dImg2 + 2 * pad2 - dFilter2) / stride2) + 1)
# creation of the output
Y = np.random.rand(n_ex, nbFilters2, output2Dim1, output2Dim2)
# creation of the second convolutionnal layer and the NN network
actFun = sigmoid
convNum2Test = ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters2, dImg1, dImg2, channels, stride1, stride2, pad1,
                                pad2)
NNconvNumtest = NeuralNetwork([convNum1Test, convNum2Test])
# gradient computation
epsilon = 0.0001

numGrad = NNconvNumtest.numGrad(Y, X_input, epsilon)
NNconvNumtest.forwardProp(X_input)
NNconvNumtest.backwardProp(Y)
Tgrad = NNconvNumtest.selfFlattenGrad()

maxError = np.max(np.abs(numGrad - Tgrad))
errorIndex = np.linalg.norm(numGrad - Tgrad) / np.linalg.norm(numGrad + Tgrad)

print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")

########## TEST ##########
print("conv to conv with id")
convNum1Test.actFun, convNum2Test.actFun = id, id
NNconvNumtest = NeuralNetwork([convNum1Test, convNum2Test])
epsilon = 0.0001

numGradTest = NNconvNumtest.numGrad(Y, X_input, epsilon)
NNconvNumtest.forwardProp(X_input)
NNconvNumtest.backwardProp(Y)
TgradTest = NNconvNumtest.selfFlattenGrad()
maxError = np.max(np.abs(numGradTest - TgradTest))
errorIndex = np.linalg.norm(numGradTest - TgradTest) / np.linalg.norm(numGradTest + TgradTest)
print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")

print("rechecking gradient of two conv layers")

actFunList = [sigmoid, sigmoid]
channels = 4
dimg1 = 15
dimg2 = 15
dimFilterList = [(3, 3), (3, 3)]
nbFilterList = [4, 3]
padList = [(0, 0), (0, 0)]
strideList = [(1, 1), (1, 1)]
convList = createConvLayers.create(actFunList, channels, dimg1, dimg2, dimFilterList, nbFilterList, padList, strideList)
n_ex = 7

X_input = np.random.rand(n_ex, channels, dimg1, dimg2)
outDim = convList[len(dimFilterList) - 1].outputDimension
Y = np.random.rand(n_ex, outDim[0], outDim[1], outDim[2])
NN2conv = NeuralNetwork(convList)
epsilon = 0.0001
numGrad = NN2conv.numGrad(Y, X_input, epsilon)
NN2conv.forwardProp(X_input)
NN2conv.backwardProp(Y)
Tgrad = NN2conv.selfFlattenGrad()
maxError = np.max(np.abs(numGrad - Tgrad))
errorIndex = np.linalg.norm(numGrad - Tgrad) / np.linalg.norm(numGrad + Tgrad)

print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")

########## FIN TEST ##########

# CONVOLUTIONNAL TO FC GRADIENT CHECK
print("checking the gradient of a convolutionnal to FC layer interface")
# filter dimensions
nbFilters = 5
dFilter1 = 3
dFilter2 = 3
# image dimensions
n_ex = 3
channels = 3
dImg1 = 15
dImg2 = 15
# creation of the input
X_input = np.random.rand(n_ex, channels, dImg1, dImg2)
# stride and padding
stride1 = 1
stride2 = 1
pad1 = 0
pad2 = 0
# computation of the output dimensions
outputDim1 = int(np.floor((dImg1 + 2 * pad1 - dFilter1) / stride1) + 1)
outputDim2 = int(np.floor((dImg2 + 2 * pad2 - dFilter2) / stride2) + 1)
# creation of the convolutionnal layer
actFun = sigmoid
conv2FCNumTest1 = ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters, dImg1, dImg2, channels, stride1, stride2,
                                   pad1, pad2)

# FC dimensions
inputSize = nbFilters * outputDim1 * outputDim2
outputSize = 12
# generation of the labels
Y = np.random.rand(n_ex, outputSize)
conv2FCNumTest2 = FullyConnectedLayer(inputSize, outputSize, sigmoid)
# creation of the NN
NNconv2FCNumtest = NeuralNetwork([conv2FCNumTest1, conv2FCNumTest2])
# gradient computation
epsilon = 0.0001
numGrad = NNconv2FCNumtest.numGrad(Y, X_input, epsilon)
NNconv2FCNumtest.forwardProp(X_input)
NNconv2FCNumtest.backwardProp(Y)
Tgrad = NNconv2FCNumtest.selfFlattenGrad()

maxError = np.max(np.abs(numGrad - Tgrad))
errorIndex = np.linalg.norm(numGrad - Tgrad) / np.linalg.norm(numGrad + Tgrad)

print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")

# CHECKING GRADIENT 3 CONV LAYERS

print("Checking gradient of three conv layers")

actFunList = [sigmoid, sigmoid, sigmoid]
channels = 4
dimg1 = 16
dimg2 = 32
dimFilterList = [(4, 4), (3, 3), (2, 2)]
nbFilterList = [12, 6, 3]
padList = [(0, 0), (0, 0), (0, 0)]
strideList = [(1, 1), (1, 1), (1, 1)]
convList = createConvLayers.create(actFunList, channels, dimg1, dimg2, dimFilterList, nbFilterList, padList, strideList)
n_ex = 7

X_input = np.random.rand(n_ex, channels, dimg1, dimg2)
outDim = convList[len(dimFilterList) - 1].outputDimension
Y = np.random.rand(n_ex, outDim[0], outDim[1], outDim[2])
NN3conv = NeuralNetwork(convList)
epsilon = 0.0001
numGrad = NN3conv.numGrad(Y, X_input, epsilon)
NN3conv.forwardProp(X_input)
NN3conv.backwardProp(Y)
Tgrad = NN3conv.selfFlattenGrad()
maxError = np.max(np.abs(numGrad - Tgrad))
errorIndex = np.linalg.norm(numGrad - Tgrad) / np.linalg.norm(numGrad + Tgrad)

print("The maximal error is " + str(maxError))
print("The error index is " + str(errorIndex) + " (this should be around or below 10e-9)")
