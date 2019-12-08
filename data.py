import mnist_loader
import neural

# def load_mnist(filename):
#     # need to unzip the files before we can read from them
#     with gzip.open(filename, 'rb') as f:
#         # each file is a binary stream of data, so we need to read it like that
#         data = np.frombuffer(f.read(), np.uint8, offset=16)
#         # reshape the images that we get into 28 x 28
#         # -1 means that we're letting the program infer what to do 
#         data = data.reshape(-1,1,28,28)
        
#         # return as a float value
#         return data/np.float32(256)

# def load_mnist_labels(filename):
#     # open file and read the labels so we can apply them
#     with gzip.open(filename, 'rb') as f:
#         data = np.frombuffer(f.read(), np.uint8, offset=8)
#         return data

# X_train = load_mnist('train-images-idx3-ubyte.gz')
# X_test = load_mnist('t10k-images-idx3-ubyte.gz')
# Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
# Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


# plt.show(plt.imshow(X_train[1][0]))
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# neural network with 4 layers:
net = neural.Network([784, 16, 16, 10])
net.training(training_data, 30, 10, 3.0, test_data=test_data)
