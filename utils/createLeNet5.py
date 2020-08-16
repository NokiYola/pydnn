from NeuralNetwork import NeuralNetwork
from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.ConvolutionLayer import ConvolutionLayer
from layers.MaxPool import MaxPool
from utils.activationFunctions import sigmoid


def createLeNet5(indputDim, regParam=0.001, regString="L2"):
    ########## FIRST LAYER ##########
    # Convolution layer
    channels, dimg1, dimg2 = indputDim
    stride1, stride2 = 1, 1
    pad1, pad2 = 0, 0
    actFun = sigmoid

    # from LeNet5 parameters:
    nbFilters = 5
    dFilter1 = 5
    dFilter2 = 5

    LeNet1Conv = ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters, dimg1, dimg2, channels, stride1, stride2, pad1,
                                  pad2)

    # the output will be of size floor((28+2*0-5)/1)+1=28-5+1=24  (output size : 5*24*24)

    ########## SECOND LAYER ##########
    # pool layer (normally average pool but max pool here)
    # from LeNet5 parameters
    size1, size2 = 2, 2
    LeNet2pool = MaxPool(size1, size2)
    # output size : 5*24/2*24/2 = 5*12*12

    ########## THIRD LAYER ##########
    # convolution layer
    # from previous layers
    channels = 5
    dimg1, dimg2 = 12, 12

    # from LeNet5 parameters:
    nbFilters = 16
    dFilter1 = 5
    dFilter2 = 5
    actFun = sigmoid

    LeNet3Conv = ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters, dimg1, dimg2, channels, stride1, stride2, pad1,
                                  pad2)

    # ouptut dimensions (12+2*0-5)//1+1=12-5+1=8 (dimensions 16*8*8)

    ########## FOURTH LAYER ##########
    # pool layer
    # from LeNet5 parameters
    size1, size2 = 2, 2

    LeNet4pool = MaxPool(size1, size2)

    # output dimensions : 16*8/2*8/2 = 16*4*4

    ########## FIFTH LAYER ##########
    # Fully connected layer
    # From previous layers
    inputDim = 256  # 16*4*4=16*16=256
    # From LeNet5 parameters
    outputDim = 120
    actFun = sigmoid

    LeNet5fc = FullyConnectedLayer(inputDim, outputDim, actFun)

    ########## SIXTH LAYER ##########
    # Fully connected layer
    # From previous layers
    inputDim = 120
    # From LeNet5 parameters
    outputDim = 84
    actFun = sigmoid

    LeNet6fc = FullyConnectedLayer(inputDim, outputDim, actFun)

    ########## SEVENTH LAYER ##########
    # Fully connected layer
    # From previous layers
    inputDim = 84
    # From LeNet5 parameters
    outputDim = 10
    actFun = sigmoid

    LeNet7fc = FullyConnectedLayer(inputDim, outputDim, actFun)

    ########## END AND RETURN ##########
    layerList = [LeNet1Conv, LeNet2pool, LeNet3Conv, LeNet4pool, LeNet5fc, LeNet6fc, LeNet7fc]
    return NeuralNetwork(layerList, regParam=regParam, regString=regString)
