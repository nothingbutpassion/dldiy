import numpy as np


class FooNet:
    def __init__(self):
        self.layers = {}
        self.input_shapes = {}
        self.output_shapes = {}
        self.params = {}
        self.grads = {}
        self.loss_func = None
        self.optimizer = None
        self.initalizer = None

    def add(self, layer, output_shape, input_shape=None):
        i = len(self.layers)
        self.layers[i] = layer
        self.output_shapes[i] = output_shape
        self.input_shapes[i] = input_shape
        if self.input_shapes[i] is None:
            self.input_shapes[i] = self.output_shapes[i-1]

    def summary(self):
        print("%s" % "-"*60)
        print("%-20s%-20s%-20s" % ('Layer (Name)','Input Shape','Output Shape'))
        print("%s" % "-"*60)
        for k,v in self.layers.items():
            print("%-20s%-20s%-20s" % (str(k)+" ("+ v.name +")" , str(self.input_shapes[k]), str(self.output_shapes[k])))
        print("%s" % "-"*60)

    def compile(self, loss_func, optimizer, initalizer):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.initalizer = initalizer
        for k in self.layers.keys():
            if self.layers[k].name == "affine":
                fan_in = self.input_shapes[k][-1]
                fan_out = self.output_shapes[k][-1]
                self.layers[k].W = self.initalizer((fan_in, fan_out))
                self.layers[k].b = self.initalizer((fan_out,))
                self.params["W" + str(k)] = self.layers[k].W
                self.params["b" + str(k)] = self.layers[k].b

    def train_one_batch(self, batch_x, batch_y):
        # forward propagation
        x = batch_x
        for v in self.layers.values():
            x = v.forward(x)

        # backward propagation
        keys = list(self.layers.keys())
        keys.reverse()
        grad = self.loss_func.grad(batch_y, x)
        for k in keys:
            grad = self.layers[k].backward(grad)

        # update grads
        for k,v in self.layers.items():
            if self.layers[k].name == "affine":
                self.grads["W" + str(k)] = self.layers[k].dW
                self.grads["b" + str(k)] = self.layers[k].db
        self.optimizer.update(self.params, self.grads)

    def train(self, train_x, train_y, batch_size, epochs=1):
        history = {"acc": [], "loss": [] }
        steps = train_x.shape[0]//batch_size
        for j in range(epochs):
            for i in range(steps):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                self.train_one_batch(batch_x, batch_y)

            # predict all train samples
            y_pred = self.predict(train_x)

            # caculate loss
            loss = self.loss_func.loss(train_y, y_pred)
            if not history["loss"] is None:
                history["loss"].append(loss)

            # caculate accuracy
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(train_y, axis=1)
            acc = np.sum(y_pred == y_true) / float(train_x.shape[0])
            if not history["acc"] is None:
                history["acc"].append(acc) 
            
            print("== epochs: " + str(j) + " loss: " + str(loss) + " acc: " + str(acc))
        
        return history
    
    def predict(self, x):
        for v in self.layers.values():
            x = v.forward(x)
        return x
    
    def loss(self, x, y_true):
        y_pred = self.predict(x)
        return self.loss_func.loss(y_true, y_pred)

    def accuracy(self, x, y_true):
        y_pred = self.predict(x)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.sum(y_pred == y_true) / float(x.shape[0])

            