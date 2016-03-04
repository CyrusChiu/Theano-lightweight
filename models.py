import sys
import os
import glob
import time
import theano
import theano.tensor as T
import numpy as np
from optimizers import SGD, Adagrad
from layers import FCLayer, DropoutLayer, SoftmaxLayer
from layers import LSTMlayer, SimpleRNNlayer, BLSTMLayer
import tools

class MLP_3layers(object):

    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config

        batch_size = config['batch_size']

        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        fc_layer1 = FCLayer(input=x, n_in=config['n_features'], n_out=2048)
        layers.append(fc_layer1)
        params += fc_layer1.params
        weight_types += fc_layer1.weight_type

        fc_layer2 = FCLayer(input=fc_layer1.output, n_in=2048, n_out=2048)
        layers.append(fc_layer2)
        params += fc_layer2.params
        weight_types += fc_layer2.weight_type

        fc_layer3 = FCLayer(input=fc_layer2.output, n_in=2048, n_out=2048)
        layers.append(fc_layer3)
        params += fc_layer3.params
        weight_types += fc_layer3.weight_type

        softmax_layer4 = SoftmaxLayer(
                            input=fc_layer3.output,
                            n_in=2048,
                            n_out=config['n_class'])
        layers.append(softmax_layer4)
        params += softmax_layer4.params
        weight_types += softmax_layer4.weight_type
        #----------------------------------

        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1]=='W']
        self.L2_sqr = sum(L2_sqr)

        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr / batch_size
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.snapshot = config['snapshot']


class MLP_3layers_dropout(object):

    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config

        batch_size = config['batch_size']

        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        fc_layer1 = FCLayer(input=x, n_in=config['n_features'], n_out=2048)
        layers.append(fc_layer1)
        params += fc_layer1.params
        weight_types += fc_layer1.weight_type

        fc_layer2 = FCLayer(input=fc_layer1.output, n_in=2048, n_out=2048)
        layers.append(fc_layer2)
        params += fc_layer2.params
        weight_types += fc_layer2.weight_type

        fc_layer3 = FCLayer(input=fc_layer2.output, n_in=2048, n_out=2048)
        layers.append(fc_layer3)
        params += fc_layer3.params
        weight_types += fc_layer3.weight_type

        dropout_layer4 = DropoutLayer(input=fc_layer3.output, n_in=2048, n_out=2048)

        softmax_layer5 = SoftmaxLayer(
                            input=dropout_layer4.output,
                            n_in=2048,
                            n_out=config['n_class'])
        layers.append(softmax_layer5)
        params += softmax_layer5.params
        weight_types += softmax_layer5.weight_type
        #----------------------------------

        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1]=='W']
        self.L2_sqr = sum(L2_sqr)

        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr / batch_size
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.snapshot = config['snapshot']

class MLP_2layers(object):

    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config

        batch_size = config['batch_size']

        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        fc_layer1 = FCLayer(input=x, n_in=config['n_features'], n_out=2048)
        layers.append(fc_layer1)
        params += fc_layer1.params
        weight_types += fc_layer1.weight_type

        fc_layer2 = FCLayer(input=fc_layer1.output, n_in=2048, n_out=2048)
        layers.append(fc_layer2)
        params += fc_layer2.params
        weight_types += fc_layer2.weight_type



        softmax_layer3 = SoftmaxLayer(
                            input=fc_layer2.output,
                            n_in=2048,
                            n_out=config['n_class'])
        layers.append(softmax_layer3)
        params += softmax_layer3.params
        weight_types += softmax_layer3.weight_type
        #----------------------------------

        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1]=='W']
        self.L2_sqr = sum(L2_sqr)

        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr / batch_size
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.snapshot = config['snapshot']

class MLP_2layers_dropout(object):

    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config

        batch_size = config['batch_size']

        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        fc_layer1 = FCLayer(input=x, n_in=config['n_features'], n_out=2048)
        layers.append(fc_layer1)
        params += fc_layer1.params
        weight_types += fc_layer1.weight_type

        fc_layer2 = FCLayer(input=fc_layer1.output, n_in=2048, n_out=2048)
        layers.append(fc_layer2)
        params += fc_layer2.params
        weight_types += fc_layer2.weight_type

        dropout_layer4 = DropoutLayer(input=fc_layer2.output, n_in=2048, n_out=2048,prob_drop = 0.5)

        softmax_layer5 = SoftmaxLayer(
                            input=dropout_layer4.output,
                            n_in=2048,
                            n_out=config['n_class'])
        layers.append(softmax_layer5)
        params += softmax_layer5.params
        weight_types += softmax_layer5.weight_type
        #----------------------------------

        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1]=='W']
        self.L2_sqr = sum(L2_sqr)

        # cost should not / batch_size and 0.5 should not be here too
        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr / batch_size
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.snapshot = config['snapshot']
        self.proba = layers[-1].p_y_given_x

class VanillaRNN(object):

    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config


        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        rnn_layer1 = SimpleRNNlayer(input=x, n_hidden=200, n_in=config['n_features'],n_out=config['n_class'])
        layers.append(rnn_layer1)
        params += rnn_layer1.params
        weight_types += rnn_layer1.weight_type
        #----------------------------------
        # L2_sqr, cost, errors y_pred
        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1] in ['Wh','Wi','Wo']]
        self.L2_sqr = sum(L2_sqr)

        # cost should not / batch_size and 0.5 should not be here too
        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.snapshot = config['snapshot']

class LSTM(object):

    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config


        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        lstm_layer1 = LSTMlayer(input=x, n_in=config['n_features'],n_out=200)
        layers.append(lstm_layer1)
        params += lstm_layer1.params
        weight_types += lstm_layer1.weight_type
        #----------------------------------
        # L2_sqr, cost, errors y_pred
        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1] in ['Wi','Ui',
                                                                                  'Wc','Uc',
                                                                                  'Wf','Uf',
                                                                                  'Wo','Uo']]
        self.L2_sqr = sum(L2_sqr)

        # cost should not / batch_size and 0.5 should not be here too
        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.snapshot = config['snapshot']

class BLSTM(object):
    def __init__(self, input=None, config=None):
        self.input = input
        self.config = config


        layers = []
        params = []
        weight_types = []

        x = T.matrix('x')
        y = T.ivector('y')

        blstm_layer1 = BLSTMLayer(input=x, n_in=config['n_features'],n_out=200)
        layers.append(blstm_layer1)
        params += blstm_layer1.params
        weight_types += blstm_layer1.weight_type
        #----------------------------------
        # L2_sqr, cost, errors y_pred
        L2_sqr = [(p[0] ** 2).sum() for p in zip(params,weight_types) if p[1] in ['Wi','Ui',
                                                                                  'Wc','Uc',
                                                                                  'Wf','Uf',
                                                                                  'Wo','Uo',
                                                                                  'Wbi','Ubi',
                                                                                  'Wbc','Ubc',
                                                                                  'Wbf','Ubf',
                                                                                  'Wbo','Ubo']]
        self.L2_sqr = sum(L2_sqr)

        # cost should not / batch_size and 0.5 should not be here too
        self.cost = layers[-1].negative_log_likelihood(y) + 0.5 * config['l2'] * self.L2_sqr
        self.errors = layers[-1].errors(y)
        self.y_pred = layers[-1].y_pred
        self.weight_types = weight_types
        self.params = params
        self.layers = layers
        self.x = x
        self.y = y
        self.snapshot = config['snapshot']
