### Requirements 

`pip install numpy`

### Usage

Download https://bit.ly/32109cs and place it in the root folder

2 ways to run it :

`python mainTestLeNet5.py`

`python mainTestOptimize.py`

`python mainTestLeNet5.py` will create a neural net very similar to LeNet5 and run it on MNIST data (optimization algorithm : minibatch gradient descent with inertia)

`python mainTestOptimize.py` will create a fully connected neural net with three hidden layers and run it on MNIST data (optimization algorithm : minibatch gradient descent with inertia)

The layer folder contains three classes, each one corresponding to a layer type (fully connected, convolution and max pool). These layers are reprogramed from scratch with numpy for vectorization

The NeuralNetwork is the class that performs the optimization. Given a list of instances of layers (which represents the net you're working on) you can perform a forward propagation, a backward propagation or launch an optimization algorithm on the whole net


### Credits

Sylvain Gugger - https://sgugger.github.io/convolution-in-depth.html#convolution-in-depth
