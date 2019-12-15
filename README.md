<h1 align="center">Drawing numbers</h1>

## By: Jerry Lee, Habib Sorathia, Calvin Ly, Wilson Zhang

[Screenshots and notes](https://docs.google.com/document/d/1bLg6yaiKLpGwETKzg33juMflCuPrBKwTKRTlUoOGR6s/edit?usp=sharing)

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
The data structure used most in our project is lists. Our neural network consists of three main lists: size, biases, weights. We also used lists in the back-end of the website. This was so we could store the information of the user generated numbers (their images), and the guessed value so the user
could view and remove them later.


# Complexity analysis
This neural network as a whole contained many functions, many of which have their own complexities. 

Let:
1. *b*: number of batches
2. *d*: size of test data
3. *e*: epoch number
4. *m*: number of layers in the neural network
5. *n*: average number of weights in each layer
6. *s*: size of a batch


## Backprop: `def backprop(self, x, y)`
This function performed several duties, including feeding the test data forward and adjusting the nodes of the neural network based on the result from the feed forward portion. The time complexity of the *feed forward* portion is ***O(mn)***. The time complexity for the *backward pass* portion is ***O(mn)***. Thus, the time complexity of this function is ***O(mn)*** Unsurprisingly, the time complexity of both functions are the same. the feed forward function and backward pass function are just different methods of stepping through the neural network. They both consist of the O(n) complexity from the dot product and O(m) complexity from traversing the layers.

## Upbatch: `def upbatch(self, batch, eta)`
This function updated the neural network's weights and biases based on the data from the batch. The time complexity of upbatch is ***O(mns)***. `Backprop` has a time complexity of O(mn) and the `for` loop executes this function once for each element of batch.

## Training: `def training(self, data, epoch, batchSize, eta, test_data=None)`
This function is called to train the neural network. The time complexity of training is ***O(mnsbe)***. Within this function, there is a loop that runs as many times as the given epoch, hence the *e*. Inside the loop, there is a `shuffle` function, which is O(d). Also within the epoch loop is a loop that passes each smaller batch into `upbatch`. The time complexity of the batch loop is O(mnsb) from the function.

## Feedforward: `def feedforward(self, a)`
Feedforward is the primary function called when a user wants to see what digit was drawn. The time complexity of feedforward is ***O(mn)***. It is the same as the feedforward in backprop.

## Evaluate: `def evaluate(self, test_data)`
Evaluate is used in the training function to determine how much of the test data worked properly. The time complexity of this function is ***O(mnd)***. It runs `feedforward` for every element within test data of size d. 

## Initializing: `def __init__(self, sizes=[784, 16, 16, 10])`
```__init__``` is the constructor of the neural network. The default size of the neural network is `[784,16,16,10]`, with 1 input layer of size 784, 2 hidden layers of size 16, and an output layer of 10. The time complexity of this function is ***O(mn)***, which occurs when setting up the weights.


# Tools used

## PIL
PIL stands for Python imaging library and is also known as Pillow. PIL comes with a variety of imaging tools, including the ones we used to scale our image to 20x20 and back up to 28x28. 

## Numpy
Numpy is a python library that comes with utilities for linear algebra. It provids more efficient methods of array manipulation. In our neural network, we used this for shuffling, randomizing, dot products, and finding the index of the maximum value.

## MNIST
MNIST provided a dataset of 10,000 handwritten digits which we used to train our neural network in. We used this in `mnist_loader.py` to return the test cases.

## Flask
We used flask, which is a package for Python that can serve as a micro-web framework. We utilzed this to serve as the controller for our website so the user could navigate and route through different web pages with ease.

## Boostrap 
We utilized bootstrap and it's dependencies to make our website look much prettier.

## Heroku
We deployed our website on heroku, and it was successfully deployed on [here](https://drawing-numbers.herokuapp.com/).

https://drawing-numbers.herokuapp.com/

## Sphinx documentation
We used sphinx documentation to document our functions and code. 

# Conclusions
This project really tested the abilities of our group. This project was a new concept for all of us since only two of us had just recently been exposed to machine learning. In the end, our website was able to work as intended, with a home, draw, and profile page. Our neural network struggled a little. It was unable to detect some (literally) edge cases where the number was drawn on the edge of the page. The digit had to be centered and even when it was, the neural network was not able to consistently detect the correct digit. 

# Contributions from each member 

### Jerry
<ul>
<li>Researched the neural network</li>
<li>Created the mnist loader and took apart the test cases</li>
<li>Created the neural network</li>
<li>Saved the neural network nodes for the website</li>
<li>Did documentation on neural network</li>
</ul>

### Calvin
<ul>
<li>Designed and created the pages for the frontend</li>
<li>Created drawing canvas for user to draw on (w/ save functionality)</li>
<li>Connected the frontend & backend to dynamically display results</li>
<li>Helped save images/outputs of the user into the database</li>
</ul>

### Habib
<ul>
<li>Created initial loader for mnist data</li>
<li>Created flask template and routes for website</li>
<li>Created login/register/logout functionality</li>
<li>Displayed cards onto profile page</li>
<li>Deployed project to herokun</li>
</ul>
### Wilson
