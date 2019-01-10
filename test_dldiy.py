import numpy as np
import matplotlib.pyplot as plt
import datasets.mnist as mnist
import layers
import losses
import optimizers
import initializers
import networks

def test_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    val_x = train_x[50000:]
    val_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
    batch_size = 200
    net = networks.BaseNet()
    net.add(layers.Affine(), (None, 28), input_shape=(None, train_x.shape[1]))  
    net.add(layers.ReLU(), (None, 28))
    net.add(layers.Affine(), (None, 10))
    net.add(layers.ReLU(), (None, 10))
    net.add(layers.Affine(), (None, 10))
    net.add(layers.Softmax(), (None, 10))
    net.summary()
    net.compile(losses.CrossEntropy(), optimizers.SGD(lr=0.001), initializers.HeNormal())
    history = net.train(train_x, train_y, batch_size, 32, val_x, val_y)
    epochs = range(1, len(history["loss"])+1)
    plt.plot(epochs, history["loss"], 'ro', label="Traning loss")
    plt.plot(epochs, history["val_loss"], 'go',label="Validating loss")
    plt.plot(epochs, history["acc"], 'r', label="Traning accuracy")
    plt.plot(epochs, history["val_acc"], 'g', label="Validating accuracy")
    plt.title('Training/Validating loss/accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show(block=True)

def test_mnist_with_cov2d():
    (train_x, train_y), (test_x, test_y) = mnist.load_data(flatten=False)
    val_x = train_x[50000:]
    val_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
    batch_size = 200
    net = networks.CovNet()
    net.add(layers.Cov2D(3, 3, stride=1, pad=1), (None, 4, 28, 28), input_shape=(None, 1, 28, 28))  
    net.add(layers.ReLU(), (None, 4, 28, 28))
    net.add(layers.MaxPooling(2, 2, stride=2), (None, 4, 14, 14))
    net.add(layers.Flatten(), (None, 4*14*14))
    net.add(layers.Affine(), (None, 10))
    net.add(layers.Softmax(), (None, 10))
    net.summary()
    net.compile(losses.CrossEntropy(), optimizers.SGD(lr=0.001), initializers.HeNormal())
    history = net.train(train_x, train_y, batch_size, 20, val_x, val_y)
    epochs = range(1, len(history["loss"])+1)
    plt.plot(epochs, history["loss"], 'ro', label="Traning loss")
    plt.plot(epochs, history["val_loss"], 'go',label="Validating loss")
    plt.plot(epochs, history["acc"], 'r', label="Traning accuracy")
    plt.plot(epochs, history["val_acc"], 'g', label="Validating accuracy")
    plt.title('Training/Validating loss/accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show(block=True)

if __name__ == "__main__":
    test_mnist_with_cov2d()
