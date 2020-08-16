import gzip
import pickle as cPickle
from utils.labelsToMat import labelsToMat
from utils.createLeNet5 import createLeNet5

# Loads the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
u = cPickle._Unpickler(f)
u.encoding = 'latin1'
train, val, test = u.load()
f.close()

# Saves the dataset in python lists
X_input = train[0].reshape((len(train[0]), 28, 28))
Y = labelsToMat(train[1], 10)

# creation of LeNet5
LeNet5 = createLeNet5((1, 28, 28))

# launches optimization algorithm
LeNet5.optimize(Y, X_input, alpha=0.6, nbElts=1000, iterMax=100, costString="MaxL", decayRate=200000)

# computes and prints the error rate on the test set
error = LeNet5.computeError(labelsToMat(test[1], 10), test[0])
print("the success rate is worth " + str(error) + "%")
