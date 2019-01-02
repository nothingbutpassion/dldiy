import numpy as np


class BaseNet:
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
            print("%-20s%-20s%-20s" % (str(k+1)+" ("+ v.name +")" , str(self.input_shapes[k]), str(self.output_shapes[k])))
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

    def train(self, train_x, train_y, batch_size, epochs=1, val_x=None, val_y=None):
        history = {"acc": [], "loss": [], "val_acc": [], "val_loss": [] }
        steps = train_x.shape[0]//batch_size
        for j in range(epochs):
            for i in range(steps):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                self.train_one_batch(batch_x, batch_y)
                #print("epochs: %-4s steps: %-20s" % (str(j+1), str(i)))

            # predict all train samples
            train_sample_x = train_x[:1000]
            train_sample_y = train_y[:1000]
            y_pred = self.predict(train_sample_x)

            # caculate training loss
            loss = self.loss_func.loss(train_sample_y, y_pred)
            history["loss"].append(loss)

            # caculate training accuracy
            acc = self.loss_func.accuracy(train_sample_y, y_pred)
            history["acc"].append(acc)

            # caculate validating loss & accuracy
            if not val_x is None and not val_y is None:
                y_pred = self.predict(val_x)
                val_loss = self.loss_func.loss(val_y, y_pred)
                val_acc = self.loss_func.accuracy(val_y, y_pred)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            
            print("epochs: %-4s loss: %-20s acc: %-20s" % (str(j+1), str(loss), str(acc)))
        
        return history
    
    def predict(self, x):
        for v in self.layers.values():
            x = v.forward(x)
        return x

    def evaluate(self, x, y_true, metrics=["loss", "acc"]):
        result = {}
        if not metrics is None:
            y_pred = self.predict(x)
            if "loss" in metrics:
                result["loss"] = self.loss_func.loss(y_true, y_pred)
            if "acc" in metrics:
                result["acc"] = self.loss_func.accuracy(y_true, y_pred)
            return result
    
    def loss(self, x, y_true):
        y_pred = self.predict(x)
        return self.loss_func.loss(y_true, y_pred)

    def accuracy(self, x, y_true):
        y_pred = self.predict(x)
        return self.loss_func.accuracy(y_true, y_pred)


class CovNet(BaseNet):
    def __init__(self):
        super().__init__()

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
            if self.layers[k].name == "cov2d":
                N, C, H, W = self.input_shapes[k]
                N, FN, OH, OW = self.output_shapes[k]
                # W.shape: (C*FH*FW, FN)  wherein, each filter is a column
                # b.shape: (FN,)
                fan_in = C * self.layers[k].FH * self.layers[k].FW
                fan_out = FN
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
            if self.layers[k].name == "affine" or self.layers[k].name == "cov2d":
                self.grads["W" + str(k)] = self.layers[k].dW
                self.grads["b" + str(k)] = self.layers[k].db
        self.optimizer.update(self.params, self.grads)
    


    

            