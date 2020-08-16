import numpy as np
from layers.ConvolutionLayer import ConvolutionLayer


def create(actFunList, channels, dimg1, dimg2, dimFilterList, nbFilterList, padList, strideList):
    layerList = []
    dFilter1, dFilter2 = dimFilterList[0]
    nbFilters = nbFilterList[0]
    actFun = actFunList[0]
    stride1, stride2 = strideList[0]
    pad1, pad2 = padList[0]

    layerList.append(
        ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters, dimg1, dimg2, channels, stride1, stride2, pad1, pad2))

    nbLayers = len(nbFilterList)
    for layerIndex in range(1, nbLayers):
        dimg1 = int(np.floor((dimg1 + 2 * pad1 - dFilter1) / stride1) + 1)
        dimg2 = int(np.floor((dimg2 + 2 * pad2 - dFilter2) / stride2) + 1)
        channels = nbFilters

        dFilter1, dFilter2 = dimFilterList[layerIndex]
        nbFilters = nbFilterList[layerIndex]
        actFun = actFunList[layerIndex]
        stride1, stride2 = strideList[layerIndex]
        pad1, pad2 = padList[layerIndex]
        layerList.append(
            ConvolutionLayer(actFun, dFilter1, dFilter2, nbFilters, dimg1, dimg2, channels, stride1, stride2, pad1,
                             pad2))

    return layerList
