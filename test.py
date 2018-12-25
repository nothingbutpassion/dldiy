import numpy as np
import datasets.mnist as mnist
import layers
import losses
import optimizers
import network

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    batch_size=200
    batch_x=train_x[:batch_size]
    batch_y=train_y[:batch_size]

    W1 = 0.01*np.random.rand(batch_x.shape[1], 100)
    b1 = 0.01*np.random.rand(100)
    W2 = 0.01*np.random.rand(100, 200)
    b2 = 0.01*np.random.rand(200)

    net = network.FooNet()
    net.add(layers.Affine(W1, b1), (batch_size, 100), input_shape=batch_x.shape)  
    net.add(layers.ReLU(), (batch_size, 100))
    net.add(layers.Affine(W2, b2), (batch_x.shape[0], 200))  
    net.add(layers.ReLU(), (batch_x.shape[0], 200)) 
    net.summary()
    net.compile(losses.CrossEntropy(), optimizers.SGD())
    net.train(batch_x, batch_y, batch_size)