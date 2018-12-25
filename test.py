import numpy as np
import datasets.mnist as mnist
import matplotlib.pyplot as plt
import layers
import losses
import optimizers
import network

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    batch_size=200
    batch_x=train_x[:batch_size]
    batch_y=train_y[:batch_size]

    scale = 0.01
    W1 = scale*np.random.rand(batch_x.shape[1], 100)
    b1 = scale*np.random.rand(100)
    W2 = scale*np.random.rand(100, 10)
    b2 = scale*np.random.rand(10)

    net = network.FooNet()
    net.add(layers.Affine(W1, b1), (batch_size, 100), input_shape=batch_x.shape)  
    net.add(layers.ReLU(), (batch_size, 100))
    net.add(layers.Affine(W2, b2), (batch_size, 10)) 
    net.add(layers.Softmax(), (batch_size, 10))
    net.summary()
    net.compile(losses.MSE(), optimizers.SGD(lr=0.005))
    loss_list, acc_list = net.train(train_x, train_y, batch_size)
    x = range(len(loss_list))
    plt.plot(x, loss_list)
    plt.plot(x, acc_list)
    plt.show()
    