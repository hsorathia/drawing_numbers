# Problem
Many older documents in an agency are handwritten. We aim to create a software tool to solve this problem.


# Solution Description
The solution to this would be image recognition, or more specifically, character recognition. This project takes a step towards character recognition by detecting digits.


# Algorithms and Data Structures

## Neural Network
A neural network is both an algorithm and data structure. Although the neural network resembles a graph, it lacks the physical nodes that are present in a graph. It has biases and weights, which are lists. It uses a back propogation algorithm to train and adjust the biases and weights. The neural network itself is also often described as an algorithm to convert images to digits. 

## Algorithms
Aside from the neural network, we also used a variety of algorithms in our website. They are documented through sphinx documentation. 

## Data structures
The data structure used most in our project is lists. Our neural network consists of three main lists: size, biases, weights. 



# Complexity analysis


# Tools used

## PIL
PIL stands for Python imaging library and is also known as Pillow. PIL comes with a variety of imaging tools, including the ones we used to scale our image to 20x20 and back up to 28x28. 

## Numpy
Numpy is a python library that comes with utilities for linear algebra. It provids more efficient methods of array manipulation. In our neural network, we used this for shuffling, randomizing, dot products, and finding the index of the maximum value.

## MNIST
MNIST provided a dataset of handwritten digits to train our neural network in. We used this in `mnist_loader.py` to return the test cases.

# Conclusions

# Contributions from each member

