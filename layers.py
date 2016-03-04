import numpy as np
import theano
import theano.tensor as T
from theano import scalar

rng = np.random.RandomState(23455)

class Weight(object):

    def __init__(self, w_shape, init=None ,mean=0, std=0.01):
        super(Weight, self).__init__()
        if init == 'he_normal':
            """  weights initialization for fclayer
            Reference:  He et al., http://arxiv.org/abs/1502.01852"""
            #print "debug:condition: he_normal"
            self.np_values = np.asarray(
                rng.randn(w_shape[0],w_shape[1]) * np.sqrt(2.0/w_shape[0]),
                    dtype=theano.config.floatX)

            self.val = theano.shared(value=self.np_values)

        elif init == 'glorot_uniform':
            """  W_in initialization for RNN-like layer
            Reference: Glorot & Bengio, AISTATS 2010 """
            scale = np.sqrt(2. / (w_shape[0] + w_shape[1]))
            self.np_values = np.asarray(
                rng.uniform(-scale, scale, w_shape),
                    dtype=theano.config.floatX)

            self.val = theano.shared(value=self.np_values)

        elif init == 'orthogonal':
            """  W/W_h or say U, initialization for RNN-like layer
            From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120 """
            flat_shape = (w_shape[0], np.prod(w_shape[1:]))
            a = rng.randn(flat_shape[0], flat_shape[1])
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            q = q.reshape(w_shape)

            self.np_values = np.asarray(1.1 * q[:w_shape[0], :w_shape[1]],
                    dtype=theano.config.floatX)
            self.val = theano.shared(value=self.np_values)

        elif init == 'zeros':
            """ bias initialization for RNN-like layer """
            self.np_values = np.asarray(np.zeros(w_shape), dtype=theano.config.floatX)
            self.val = theano.shared(value=self.np_values)

        elif init == 'ones':
            """ bias initialization for RNN-like layer
             Reference: Jozefowicz, et al. (2015) """
            self.np_values = np.asarray(np.ones(w_shape), dtype=theano.config.floatX)
            self.val = theano.shared(value=self.np_values)
        else:
            """ if init is not setting """

            if std != 0:
                self.np_values = np.asarray(
                    rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
            else:
                self.np_values = np.cast[theano.config.floatX](
                    mean * np.ones(w_shape, dtype=theano.config.floatX))

            self.val = theano.shared(value=self.np_values)

    def save_weight(self, dir, name):
        #print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)

    def load_weight_half(self, dir, name):
        print 'weight loaded: ' + name
        np_values = np.load(dir + name + '.npy')
        self.np_values  = 0.5 * np_values # 0.5 is P_drop
        self.val.set_value(self.np_values)


class SimpleRNNlayer(object):
    def __init__(self, input, n_hidden, n_in, n_out):

        #self.input = input
        # recurrent weights
        self.Wh = Weight((n_hidden, n_hidden), init='orthogonal')
        # W_in: input to hidden layer weights
        self.Wi = Weight((n_in, n_hidden), init='glorot_uniform')
        # W_out: hidden to output layer weights
        self.Wo = Weight((n_hidden, n_out), init='glorot_uniform')

        # hidden state, init with zeros
        self.h0 = Weight((n_hidden,), init='zeros') #h0
        # bias of W_hidden
        self.bh = Weight((n_hidden,), init='zeros') #bh
        # bias of W_out
        self.by = Weight((n_out,), init='zeros') #by

        self.params = [self.Wh.val, self.Wi.val, self.Wo.val,
                       self.h0.val, self.bh.val, self.by.val]
        self.weight_type = ['Wh', 'Wi', 'Wo', 'h0', 'bh', 'by']
        print 'recurrent layer with num_in: {} num_hidden: {} num_out: {}'.format(n_in,n_hidden,n_out)

        def step(x_t, a_tm1):
            a_t = sigmoid( T.dot(x_t, self.Wi.val) + T.dot(a_tm1, self.Wh.val) + self.bh.val )
            y_t = T.dot(a_t, self.Wo.val) + self.by.val
            #y_t = softmax(T.dot(a_t, self.Wo.val) + self.by.val)
            return a_t, y_t

        [self.a_seq, self.y_seq], _ = theano.scan(step,
                                               sequences=input, #x_t
                                               outputs_info=[self.h0.val, None], #a_t
                                               n_steps=input.shape[0])
        #self.y_seq_last = self.y_seq[-1][0]
        self.p_y_given_x = softmax(self.y_seq)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LSTMlayer(object):
    def __init__(self, input, n_in, n_out):

        # input gate
        self.Wi = Weight((n_in,n_out), init='glorot_uniform')
        self.Ui = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bi = Weight((n_out,), init='zeros')

        # forget gate
        self.Wf = Weight((n_in,n_out), init='glorot_uniform')
        self.Uf = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bf = Weight((n_out,), init='ones') # forget gate init is different

        # input to cell
        self.Wc = Weight((n_in,n_out), init='glorot_uniform')
        self.Uc = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bc = Weight((n_out,), init='zeros')

        # output gate
        self.Wo = Weight((n_in,n_out), init='glorot_uniform')
        self.Uo = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bo = Weight((n_out,), init='zeros')

        self.params = [
            self.Wi.val, self.Ui.val, self.bi.val,
            self.Wc.val, self.Uc.val, self.bc.val,
            self.Wf.val, self.Uf.val, self.bf.val,
            self.Wo.val, self.Uo.val, self.bo.val]

        self.weight_type = ['Wi','Ui','bi',
                            'Wc','Uc','bc',
                            'Wf','Uf','bf',
                            'Wo','Uo','bo']
        print 'LSTM layer with num_in: {} num_out: {}'.format(n_in,n_out)

        def step(x_t,
                 h_tm1, c_tm1,
                 u_i, u_f, u_o, u_c):
            i_t = hard_sigmoid( T.dot(x_t,self.Wi.val) + T.dot(h_tm1,u_i) + self.bi.val)
            f_t = hard_sigmoid( T.dot(x_t,self.Wf.val) + T.dot(h_tm1,u_f) + self.bf.val)
            c_t = f_t * c_tm1 + i_t * T.tanh( T.dot(x_t,self.Wc.val) + T.dot(h_tm1,u_c) + self.bc.val)
            o_t = hard_sigmoid( T.dot(x_t,self.Wo.val) + T.dot(h_tm1,u_o) + self.bo.val)
            #h_t = o_t * T.tanh(c_t)
            h_t = softmax(o_t * T.tanh(c_t))
            return h_t, c_t #h_t:output, c_t:memory

        [self.h_t, self.c_t], _ = theano.scan(step,
                                            sequences=input,
                                            outputs_info=[ T.alloc(np.asarray(0.,dtype=theano.config.floatX),n_in,n_out),
                                                           T.alloc(np.asarray(0.,dtype=theano.config.floatX),n_in,n_out)],
                                            non_sequences=[self.Ui.val,self.Uf.val,self.Uo.val,self.Uc.val])

        self.p_y_given_x = self.h_t[:,0,:]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class BLSTMLayer(object):
    def __init__(self, input, n_in, n_out):
        # forward weight
        # input gate
        self.Wi = Weight((n_in,n_out), init='glorot_uniform')
        self.Ui = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bi = Weight((n_out,), init='zeros')

        # forget gate
        self.Wf = Weight((n_in,n_out), init='glorot_uniform')
        self.Uf = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bf = Weight((n_out,), init='ones') # forget gate init is different

        # input to cell
        self.Wc = Weight((n_in,n_out), init='glorot_uniform')
        self.Uc = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bc = Weight((n_out,), init='zeros')

        # output gate
        self.Wo = Weight((n_in,n_out), init='glorot_uniform')
        self.Uo = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bo = Weight((n_out,), init='zeros')

        # backward weights
        # input gate
        self.Wbi = Weight((n_in,n_out), init='glorot_uniform')
        self.Ubi = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bbi = Weight((n_out,), init='zeros')

        # forget gate
        self.Wbf = Weight((n_in,n_out), init='glorot_uniform')
        self.Ubf = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bbf = Weight((n_out,), init='ones') # forget gate init is different

        # input to cell
        self.Wbc = Weight((n_in,n_out), init='glorot_uniform')
        self.Ubc = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bbc = Weight((n_out,), init='zeros')

        # output gate
        self.Wbo = Weight((n_in,n_out), init='glorot_uniform')
        self.Ubo = Weight((n_out,n_out), init='orthogonal') #hidden state
        self.bbo = Weight((n_out,), init='zeros')

        self.params = [
            self.Wi.val, self.Ui.val, self.bi.val,
            self.Wc.val, self.Uc.val, self.bc.val,
            self.Wf.val, self.Uf.val, self.bf.val,
            self.Wo.val, self.Uo.val, self.bo.val,

            self.Wbi.val, self.Ubi.val, self.bbi.val,
            self.Wbc.val, self.Ubc.val, self.bbc.val,
            self.Wbf.val, self.Ubf.val, self.bbf.val,
            self.Wbo.val, self.Ubo.val, self.bbo.val]

        self.weight_type = ['Wi','Ui','bi',
                            'Wc','Uc','bc',
                            'Wf','Uf','bf',
                            'Wo','Uo','bo',
                            'Wbi','Ubi','bbi',
                            'Wbc','Ubc','bbc',
                            'Wbf','Ubf','bbf',
                            'Wbo','Ubo','bbo']

        print 'BLSTM layer with num_in: {} num_out: {}'.format(n_in,n_out)

        def forward_step(x_t,
                         h_tm1, c_tm1,
                         u_i, u_f, u_o, u_c):
            i_t = hard_sigmoid( T.dot(x_t,self.Wi.val) + T.dot(h_tm1,u_i) + self.bi.val)
            f_t = hard_sigmoid( T.dot(x_t,self.Wf.val) + T.dot(h_tm1,u_f) + self.bf.val)
            c_t = f_t * c_tm1 + i_t * T.tanh( T.dot(x_t,self.Wc.val) + T.dot(h_tm1,u_c) + self.bc.val)
            o_t = hard_sigmoid( T.dot(x_t,self.Wo.val) + T.dot(h_tm1,u_o) + self.bo.val)
            #h_t = o_t * T.tanh(c_t)
            h_t = softmax(o_t * T.tanh(c_t))
            return h_t, c_t #h_t:output, c_t:memory

        [self.h_t, self.c_t], _ = theano.scan(forward_step,
                                            sequences=input,
                                            outputs_info=[ T.alloc(np.asarray(0.,dtype=theano.config.floatX),n_in,n_out),
                                                           T.alloc(np.asarray(0.,dtype=theano.config.floatX),n_in,n_out)],
                                            non_sequences=[self.Ui.val,self.Uf.val,self.Uo.val,self.Uc.val])

        def backward_step(x_t,
                          h_tm1, c_tm1,
                          u_i, u_f, u_o, u_c):
            i_t = hard_sigmoid( T.dot(x_t,self.Wbi.val) + T.dot(h_tm1,u_i) + self.bbi.val)
            f_t = hard_sigmoid( T.dot(x_t,self.Wbf.val) + T.dot(h_tm1,u_f) + self.bbf.val)
            cb_t = f_t * c_tm1 + i_t * T.tanh( T.dot(x_t,self.Wbc.val) + T.dot(h_tm1,u_c) + self.bbc.val)
            o_t = hard_sigmoid( T.dot(x_t,self.Wbo.val) + T.dot(h_tm1,u_o) + self.bbo.val)
            #hb_t = o_t * T.tanh(cb_t)
            hb_t = softmax(o_t * T.tanh(cb_t))
            return hb_t, cb_t #h_t:output, c_t:memory

        [self.hb_t, self.cb_t], _ = theano.scan(backward_step,
                                            sequences=input,
                                            outputs_info=[ T.alloc(np.asarray(0.,dtype=theano.config.floatX),n_in,n_out),
                                                           T.alloc(np.asarray(0.,dtype=theano.config.floatX),n_in,n_out)],
                                            non_sequences=[self.Ubi.val,self.Ubf.val,self.Ubo.val,self.Ubc.val])

        #self.output = self.h_t[0] + self.hb_t[0]
        #self.p_y_given_x = softmax(self.h_t[0]) + softmax(self.hb_t[0])
        self.p_y_given_x = self.h_t[:,0,:] + self.hb_t[:,0,:]
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



class DropoutLayer(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.5):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * input + \
            self.flag_off * self.prob_keep * input

        DropoutLayer.layers.append(self)

        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)


    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)


class FCLayer(object):

    def __init__(self, input, n_in, n_out, activation='ReLU'):

        #self.W = Weight((n_in, n_out), std=0.005)
        self.W = Weight((n_in, n_out), init='he_normal')
        self.b = Weight(n_out, mean=0.1, std=0)
        self.input = input
        lin_output = T.dot(self.input, self.W.val) + self.b.val

        if activation == 'ReLU':
            self.output = T.maximum(lin_output, 0.0)
        elif activation == 'Tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = sigmoid(lin_output)
        else:
            raise Exception(
                    "Invalid activation function:" + str(activation))

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        print 'fc layer with num_in: ' + str(n_in) + ' num_out: ' + str(n_out)


class SoftmaxLayer(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out))
        self.b = Weight((n_out,), std=0)

        self.p_y_given_x = softmax(
            T.dot(input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'softmax layer with num_in: ' + str(n_in) + \
            ' num_out: ' + str(n_out)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    out = e_x / e_x.sum(axis=1, keepdims=True)
    return out

def hard_sigmoid(x):
    out_dtype = scalar.upgrade_to_float(scalar.Scalar(dtype=x.dtype))[0].dtype
    slope = T.constant(0.2, dtype=out_dtype)
    shift = T.constant(0.5, dtype=out_dtype)
    x = (x * slope) + shift
    x = T.clip(x, 0, 1)
    return x

def sigmoid(x):
    return 1/(1+T.exp(-x))
