import theano
import theano.tensor as T
import numpy as np

class SGD(object):
    def __init__(self,lr=0.01,decay=0.,momentum=0.9):
        self.iterations = theano.shared(np.cast[theano.config.floatX](0))
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.decay = theano.shared(np.cast[theano.config.floatX](decay))
        self.momentum = theano.shared(np.cast[theano.config.floatX](momentum))
    def get_updates(self,params,cost):
        grads = T.grad(cost, params)
        self.updates = []
        for param_i, grad_i in zip(params,grads):
            m = theano.shared(np.asarray(np.zeros(param_i.get_value().shape),dtype=theano.config.floatX))
            v = self.momentum * m - self.lr * grad_i
            self.updates.append((m,v))
            new_p = (1-self.decay*self.lr)*param_i+v
            self.updates.append((param_i, new_p ))
        return self.updates

class Adagrad(object):
    def __init__(self,lr=0.01,rho=0.9,decay=0.,epsilon=1e-6):
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.epsilon = epsilon
        self.rho = theano.shared(np.cast[theano.config.floatX](rho))
        self.decay = theano.shared(np.cast[theano.config.floatX](decay))
    def get_updates(self,params,cost):
        grads = T.grad(cost, params)
        accumulators = [theano.shared(np.asarray(np.zeros(p.get_value().shape),dtype=theano.config.floatX)) for p in params]
        self.updates = []
        for param_i, grad_i,acc_i in zip(params,grads,accumulators):
            new_a = acc_i + grad_i ** 2
            self.updates.append((acc_i, new_a))
            new_p = (1-self.decay*self.lr)*param_i - self.lr*grad_i/T.sqrt(new_a+self.epsilon)
            self.updates.append((param_i, new_p))
        return self.updates


class RMSprop(object):
    def __init__(self, lr=0.001,rho=0.9, epsilon=1e-6, clip=False):
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.epsilon = epsilon
        self.rho = theano.shared(np.cast[theano.config.floatX](rho))
        self.clip = clip
    def get_updates(self, params, cost):
        grads = T.grad(cost, params)
        accumulators = [theano.shared(np.asarray(np.zeros(p.get_value().shape),dtype=theano.config.floatX)) for p in params]
        if self.clip == False:
            self.updates = []
            for param_i, grad_i, acc_i in zip(params, grads, accumulators):
                new_a = self.rho * acc_i + (1 - self.rho) * grad_i ** 2
                self.updates.append((acc_i, new_a))
                new_p = param_i - self.lr * grad_i / T.sqrt(new_a + self.epsilon)
                self.updates.append((param_i, new_p))
            return self.updates

        elif self.clip == True:
            self.updates = []
            for param_i, grad_i, acc_i in zip(params, grads, accumulators):
                new_a = self.rho * acc_i + (1 - self.rho) * T.clip(grad_i, -1., 1.) ** 2
                self.updates.append((acc_i, new_a))
                new_p = param_i - self.lr * T.clip(grad_i,-1.,1.) / T.sqrt(new_a + self.epsilon)
                self.updates.append((param_i, new_p))
            return self.updates

        else:
            raise ValueError
