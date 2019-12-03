#BALJEETFORPRESIDENT

import random

import numpy as np

class Network(object):
    def __init__(self, sizes):
        # sizes = number of sizes in each layer of the network. in this case it would be [784, 16, 16, 10]
        self.num_layers = len(sizes)
        self.sizes = sizes
        # initially all biases are randomized
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # initially all weights (nodes) are randomized
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        # a is the input of the network (the image)
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a
    
    def training(self, data, epoch, batchSize, eta, test_data=None):
        # data = training data
        # epoch = how many times to put the whole data set into the network
        # batchSize = we will splice the training data into batches and feed them in. This is the size
        # eta = learning rate
        n_test = len(test_data)
        n = len(data)
        for i in range(epoch):
            # shuffle training data to reduce potential bias
            random.shuffle(data)
            # creates the batches of data of size batchSize
            batches = [
                data[j:j+batchSize]
                for j in range(0,n,batchSize)
            ]
            for batch in batches:
                # update each batch
                self.upbatch(batch, eta)
            if test_data:
                # Print Epoch #, how many correct, number tests
                print("Epoch{0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch{0} complete".format(j))

    def upbatch(self, batch, eta):
        # batch is the data
        # eta is learning rate
        # gradient vectors of bias and weight for adjustment
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for (x,y) in batch:
            dgb, dgw = self.backprop(x, y)
            # adjust the gradient based on backprop
            gradient_b = [nb+dnb for nb, dnb in zip(gradient_b, dgb)]
            gradient_w = [nw+dnw for nw, dnw in zip(gradient_w, dgw)]
    
    def backprop(self, x, y):
        #gradient vectors of bias and weight
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        #feeding things forward
        act = x     # currently activated
        acts = [x]  # list for all activations
        z = []      # all z vectors