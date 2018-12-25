import numpy as np


class FooNet:
    def __init__(self):
        self.layers = {}
        self.input_shapes = {}
        self.output_shapes = {}
        self.params = {}
        self.grads = {}
        self.L = None
        self.optimizer = None

    def add(self, layer, output_shape, input_shape=None):
        idx = len(self.layers)
        self.layers[idx] = layer
        self.output_shapes[idx] = output_shape
        if input_shape is None:
            self.input_shapes[idx] = self.output_shapes[idx-1] 
        else:
            self.input_shapes[idx] = input_shape

    def summary(self):
        print('FooNet info:')
        for k,v in self.layers.items():
            print('layer=' + str(k)  + ' name=' + v.name 
                + ' input_shape=' + str(self.input_shapes[k])
                + ' output_shape=' + str(self.output_shapes[k]))


    def compile(self, loss, optimizer):
        self.L = loss
        self.optimizer = optimizer
        for k,v in self.layers.items():
            if self.layers[k].name == 'affine':
                self.params['W' + str(k)] = self.layers[k].W
                self.params['b' + str(k)] = self.layers[k].b

    def train_one_batch(self, batch_x, batch_y, loss_list=None, acc_list=None):
        x = batch_x
        for v in self.layers.values():
            x = v.forward(x)

        l = self.L.loss(batch_y, x)
        print('loss: ' + str(l))
        if not loss_list is None:
            loss_list.append(l)

        acc = self.accuracy(batch_x, batch_y)
        print('acc: ' + str(acc))
        if not acc_list is None:
            acc_list.append(acc) 

        g = self.L.grad(batch_y, x)
        keys = list(self.layers.keys())
        keys.reverse()
        for k in keys:
            g = self.layers[k].backward(g)

        for k,v in self.layers.items():
            if self.layers[k].name == 'affine':
                self.grads['W' + str(k)] = self.layers[k].dW
                self.grads['b' + str(k)] = self.layers[k].db

        self.optimizer.update(self.params, self.grads)

    def train(self, train_x, train_y, batch_size):
        steps = train_x.shape[0]//batch_size
        loss_list = []
        acc_list = []
        for i in range(steps):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_y = train_y[i*batch_size:(i+1)*batch_size]
            self.train_one_batch(batch_x, batch_y, loss_list, acc_list)
        return loss_list, acc_list
    
    def predict(self, x):
        for v in self.layers.values():
            x = v.forward(x)
        return x
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

            