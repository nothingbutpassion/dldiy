import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras import losses

def one_hot(indices, num_classes):
    T = np.zeros(indices.shape + (10,))
    for i, row in enumerate(T):
        row[indices[i]] = 1
    return T


def test_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    val_x = train_x[50000:]
    val_y = one_hot(train_y[50000:], 10)
    train_x = train_x[:50000]
    train_y =one_hot(train_y[:50000], 10)

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(28, activation="relu"))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer=optimizers.SGD(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    history = model.fit(train_x, train_y, batch_size=200, epochs=32, validation_data=(val_x, val_y)).history
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
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    val_x = train_x[50000:]
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1], val_x.shape[2])
    val_y = one_hot(train_y[50000:], 10)
    train_x = train_x[:50000]
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
    train_y =one_hot(train_y[:50000], 10)

    model = models.Sequential()
    model.add(layers.Conv2D(4, (3, 3), strides=(1,1), padding='same', data_format="channels_first", activation="relu", input_shape=(1, 28, 28)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), data_format="channels_first"))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer=optimizers.SGD(lr=0.001), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    history = model.fit(train_x, train_y, batch_size=200, epochs=20, validation_data=(val_x, val_y)).history
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
