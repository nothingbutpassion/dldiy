import numpy as np
import matplotlib.pyplot as plt
import datasets.mnist as mnist
import layers
import losses
import optimizers
import initializers
import models

def test_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    val_x = train_x[50000:]
    val_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
    batch_size = 200
    modle = models.Sequential()
    modle.add(layers.Linear(28, input_shape=(None, train_x.shape[1]))) 
    modle.add(layers.ReLU())
    modle.add(layers.Linear(10))
    modle.add(layers.ReLU())
    modle.add(layers.Linear(10))
    modle.add(layers.Softmax())
    modle.compile(losses.CrossEntropy(), optimizers.SGD(lr=0.001))
    modle.summary()
    history = modle.train(train_x, train_y, batch_size, epochs=32, validation_data=(val_x, val_y))
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
    modle = models.Sequential()
    modle.add(layers.Conv2D(4, (3, 3), stride=1, pad=1, input_shape=(None, 1, 28, 28))) 
    modle.add(layers.ReLU())
    modle.add(layers.MaxPooling2D((2, 2), stride=2))
    modle.add(layers.Flatten())
    modle.add(layers.Linear(10))
    modle.add(layers.Softmax())
    modle.compile(losses.CrossEntropy(), optimizers.SGD(lr=0.001))
    modle.summary()
    history = modle.train(train_x, train_y, batch_size, epochs=32, validation_data=(val_x, val_y))
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
