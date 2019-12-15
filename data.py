import mnist_loader
import neural
import numpy as np
import pickle
from PIL import Image

### retrieve data
# load sizes
fsizes = open('nn_sizes.pkl', 'rb')
sizes = pickle.load(fsizes)
fsizes.close()

# load biases
fbiases = open('nn_biases.pkl', 'rb')
biases = pickle.load(fbiases)
fbiases.close()

# load weights
fweights = open('nn_weights.pkl', 'rb')
weights = pickle.load(fweights)
fweights.close()

# set up neural network based on file
net = neural.Network(sizes=sizes)
net.loadNodes(biases=biases, weights=weights)
def prepImage():
    """
    Turns image from png to a 2d grayscale array
    """
    # open image
    img = Image.open("result.png", 'r')
    # img.show()
    # resize image
    img = img.resize((20, 20))
    img = img.resize((28, 28), Image.ANTIALIAS)
    # make grayscale
    img = img.convert('L')
    # img.show()
    arr = np.array(img)
    # print(arr)
    return arr

# loaing training data. uncomment to train
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# Training neural network with 4 layers, uncomment to train
# net = neural.Network(sizes=[784, 16, 16, 10])
# net.training(training_data, 30, 10, 3.0, test_data=test_data)

# running neural network
arr = prepImage()
# convert arr to 1d
arr = np.reshape(arr, (784,1))
arr = (255-arr)/256
result = (net.feedforward(arr))
[print(i, "|", x) for i,x in enumerate(result)]
print ("result:", np.argmax(result))



# store data after training, uncomment to store
sizes, biases, weights = net.getNetwork()

### write to separate files, uncomment to write
# fsizes = open('nn_sizes.pkl', 'wb')
# pickle.dump(sizes, fsizes)
# fsizes.close()

# fbiases = open('nn_biases.pkl', 'wb')
# pickle.dump(biases, fbiases)
# fbiases.close()

# fweights = open('nn_weights.pkl', 'wb')
# pickle.dump(weights, fweights)
# fweights.close()




# with open('net.txt', 'w') as f:
#     f.write('sizes: ' + str(sizes) + 'ENDSIZE\n')
#     f.write('biases: ' + str(biases) + 'ENDBIAS\n')
#     f.write('weights: ' + str(weights) + 'ENDWEIGHTS\n')