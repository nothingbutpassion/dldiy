from __future__ import print_function
import numpy as np
import sys

def _print_progress(epoch, total_steps, current_step):
    pos = int(round(40.0*current_step/total_steps))
    progress = "="*pos + "-"*(40-pos)
    sys.stdout.write("Epoch: %-4s %-7s[%s]\r" % (epoch, str(current_step)+"/"+str(total_steps), progress))
    sys.stdout.flush()

class Sequential(object):
    def __init__(self):
        self.layers = {}
        self.params = {}
        self.grads = {}
        self.loss = None
        self.metrics = None
        self.optimizer = None

    def add(self, layer):
        self.layers[len(self.layers)] = layer

    def summary(self):
        print("%s" % "-"*60)
        print("%-20s%-20s%-20s" % ('Layer (Name)','Input Shape','Output Shape'))
        print("%s" % "-"*60)
        for i,v in self.layers.items():
            print("%-20s%-20s%-20s" % (str(i+1)+" ("+ v.name +")" , str(v.input_shape), str(v.output_shape)))
        print("%s" % "-"*60)

    def compile(self, loss, optimizer, metrics=[]):
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        for i, layer in self.layers.items():
            if layer.input_shape:
                layer.build(layer.input_shape)
            else:
                layer.build(self.layers[i-1].output_shape)
            if hasattr(layer, "W") and hasattr(layer, "b"):
                self.params["W" + str(i)] = layer.W
                self.params["b" + str(i)] = layer.b

    def train_one_batch(self, batch_x, batch_y):
        # forward propagation
        x = batch_x
        for layer in self.layers.values():
            x = layer.forward(x)

        # backward propagation
        keys = list(self.layers.keys())
        keys.reverse()
        grad = self.loss.grad(batch_y, x)
        for k in keys:
            grad = self.layers[k].backward(grad)

        # update grads
        for i, layer in self.layers.items():
            if hasattr(layer, "dW") and hasattr(layer, "db"):
                self.grads["W" + str(i)] = layer.dW
                self.grads["b" + str(i)] = layer.db
        self.optimizer.update(self.params, self.grads)

    def train(self, train_x, train_y, batch_size, epochs=1, validation_data=None):
        history = {"loss": [], "val_loss": []}
        for m in self.metrics:
            history[m.__name__] = []
            history["val_" + m.__name__] =[]

        steps = train_x.shape[0]//batch_size
        for j in range(epochs):
            for i in range(steps):
                batch_x = train_x[i*batch_size:(i+1)*batch_size]
                batch_y = train_y[i*batch_size:(i+1)*batch_size]
                self.train_one_batch(batch_x, batch_y)
                _print_progress(j+1, steps, i+1)

            # predict all train samples (random batch samples)
            batch_indexes = np.arange(train_x.shape[0])
            np.random.shuffle(batch_indexes)
            train_sample_x = train_x[batch_indexes[:batch_size]]
            train_sample_y = train_y[batch_indexes[:batch_size]]
            y_pred = self.predict(train_sample_x)

            # caculate training loss
            loss = self.loss.loss(train_sample_y, y_pred)
            history["loss"].append(loss)

            # caculate training metrics
            metric = {}
            for m in self.metrics:
                history[m.__name__].append(m(train_sample_y, y_pred))
                metric[m.__name__] = history[m.__name__][-1]

            # caculate validating loss & accuracy
            if validation_data:
                val_x, val_y = validation_data
                y_pred = self.predict(val_x)
                val_loss = self.loss.loss(val_y, y_pred)
                history["val_loss"].append(val_loss)
                for m in self.metrics:
                    history["val_" + m.__name__].append(m(val_y, y_pred))
            
            print("Epoch: %-4s loss: %-20s metrics: %-20s" % (str(j+1), str(loss), str(metric)))
        
        return history
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def evaluate(self, x, y_true):
        result = {}
        y_pred = self.predict(x)
        for m in self.metrics:
            result[m.__name__] = m(y_true, y_pred)
        return result

    def compute_loss(self, x, y_true):
        y_pred = self.predict(x)
        return self.loss.loss(y_true, y_pred)

    

            