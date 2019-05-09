import numpy as np
import matplotlib.pyplot as plt
import datasets.mnist as mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses


def test_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    val_x = train_x[50000:]
    val_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
    model = models.Sequential()
    model.add(layers.Dense(28, activation="relu", input_shape=(train_x.shape[1],)))
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
    (train_x, train_y), (test_x, test_y) = mnist.load_data(flatten=False)
    val_x = train_x[50000:]
    val_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
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
