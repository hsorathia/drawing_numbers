import random
import numpy as np


class Network:
    def __init__(self, sizes=[784, 16, 16, 10]):
        """
        Initialize the neural network with the given layers. The neural network has a total of 4 layers.

        Layer 1:
            784 nodes corresponding to the grayscale values for pixels in the 28x28 image
        Layers 2 and 3:
            Hidden layers of the neural network that are modified from training
        Layer 4:
            Output layer of the confidence the network has in each digit

        """
        # sizes = number of sizes in each layer of the network. in this case it would be [784, 16, 16, 10]
        self.num_layers = len(sizes)  # integer
        self.sizes = sizes  # array of integers
        # initially all biases are randomized
        # self.biases = biases if biases else [np.random.randn(y,1) for y in sizes[1:]] # array of floats
        self.biases = [np.random.randn(y, 1)
                       for y in sizes[1:]]  # array of floats
        # fix bias loading
        # initially all weights (nodes) are randomized
        # self.weights = weights if weights else [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])] # array of floats
        self.weights = [np.random.randn(y, x) for x, y in zip(
            sizes[:-1], sizes[1:])]  # array of floats
        print("Weights and bias set up")

    def getNetwork(self):
        """
        Return the sizes, biases, and weights after the network has been trained
        """
        return (self.sizes, self.biases, self.weights)

    def loadNodes(self, biases, weights):
        """
        Initialize the nodes from a file; could be combined with __init__
        """
        self.biases = biases
        self.weights = weights

    def feedforward(self, a):
        """
        Feed forward the inputs to receive the outputs from the output node
        """
        # a is the input of the network (the image)
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def training(self, data, epoch, batchSize, eta, test_data=None):
        """
        Used to train the neural network
        """
        # data = training data
        # epoch = how many times to put the whole data set into the network
        # batchSize = we will splice the training data into batches and feed them in. This is the size
        # eta = learning rate

        # convert data/test_data to lists and find length
        data = list(data)
        test_data = list(test_data)
        n_test = len(test_data)
        n = len(data)

        for i in range(epoch):
            # shuffle training data to reduce potential bias
            random.shuffle(data)
            # creates the batches of data of size batchSize
            batches = [
                data[j:j+batchSize]
                for j in range(0, n, batchSize)
            ]
            for batch in batches:
                # update each batch
                self.upbatch(batch, eta)
            if test_data:
                # Print Epoch #, how many correct, number tests
                print("Epoch{}: {} / {}".format(i,
                                                self.evaluate(test_data), n_test))
            else:
                print("Epoch{} complete".format(i))

    def upbatch(self, batch, eta):
        """
        Updates the neural network's weights and biases
        """
        # batch is the data
        # eta is learning rate
        # gradient vectors of bias and weight for adjustment
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in batch:
            dgb, dgw = self.backprop(x, y)
            # adjust the gradient based on backprop
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, dgb)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, dgw)]
        self.weights = [w-(eta/len(batch))*nw
                        for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b-(eta/len(batch))*nb
                       for b, nb in zip(self.biases, gradient_b)]

    def backprop(self, x, y):
        """
        Returns a tuple that represents the gradient for the cost function
        """
        # gradient vectors of bias and weight
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # feeding things forward
        activation = x     # currently activated
        activations = [x]  # list for all activations
        zs = []            # all z vectors
        # check every bias/weight
        for b, w in zip(self.biases, self.weights):
            # dot product between weight and activation layer
            z = np.dot(w, activation) + b
            zs.append(z)
            # feed z through the sigmoid function
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # calculate steepest descent
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])
        # update gradient
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())
        # renumbering each neuron
        for x in range(2, self.num_layers):
            z = zs[-x]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-x+1].transpose(), delta) * sp
            gradient_b[-x] = delta
            gradient_w[-x] = np.dot(delta, activations[-x-1].transpose())
        return (gradient_b, gradient_w)

    # how many test inputs output the correct result
    def evaluate(self, test_data):
        """
        Returns a number that represents how many test inputs output the correct results
        """
        # obtain results by feeding everything forward. results[0] is network result, results[1] is expected
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

    # return vector of steepest descent from partial derivatives
    def cost_derivative(self, out_act, y):
        """
        Returns vector of steepest descent from partial derivatives
        """
        return (out_act-y)

# sigmoid function (copied from github)


def sigmoid(z):
    """
    Sigmoid function
    """
    return 1.0/(1.0+np.exp(-z))

# derivative of sigmoid (copied from github because calculus is hard)


def sigmoid_prime(z):
    """
    Derivative of sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))
