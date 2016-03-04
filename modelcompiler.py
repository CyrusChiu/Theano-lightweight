import sys
import os
import glob
import time
import theano
import theano.tensor as T
import numpy as np
from optimizers import SGD, Adagrad, RMSprop
from layers import FCLayer, DropoutLayer, SoftmaxLayer
from layers import LSTMlayer, SimpleRNNlayer
import tools



class ModelCompiler(object):
    def __init__(self,model,config, optimizer='SGD'):
        self.model = model(config=config)
        if optimizer == 'SGD':
            self.optimizer = SGD(lr=config['learning_rate'],
                                decay=config['weight_decay'],
                                momentum=config['momentum'])
        elif optimizer =='Adagrad':
            self.optimizer = Adagrad(lr=config['learning_rate'], decay=config['weight_decay'])
        elif optimizer =='RMSprop':
            self.optimizer = RMSprop(lr=config['learning_rate'])

        self.config = config

    def share_var(self, data_xy, testing=False, borrow=True):
        if testing:
            assert type(data_xy) == np.ndarray, "using test data in testing step"
            shared_x = theano.shared(np.asarray(data_xy,dtype=theano.config.floatX),borrow=borrow)
            return shared_x
        else: # training
            assert type(data_xy) == tuple, "label data was missing or something else"
            data_x, data_y = data_xy
            shared_x = theano.tensor._shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
            shared_y = theano.tensor._shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
            return shared_x, T.cast(shared_y,'int32')

    def _train_by_sentence_init_(self, x_train, y_train, x_val, y_val, l_t, l_v):
        #x_train, y_train, x_val, y_val = [], [], [], []
        #for each in train:
        #    x_train.extend(each['data'].tolist())
        #    y_train.extend(each['label'].tolist())
        #for each in val:
        #    x_val.extend(each['data'].tolist())
        #    y_val.extend(each['label'].tolist())

        #x_train = np.asarray(x_train).astype('float32')
        #y_train = np.asarray(y_train).astype('int32')
        #x_val = np.asarray(x_val).astype('float32')
        #y_val = np.asarray(y_val).astype('int32')

        self.learning_rate_decay = self.config['learning_rate_decay']
        train_set_x, train_set_y = self.share_var((x_train,y_train))
        valid_set_x, valid_set_y = self.share_var((x_val,y_val))

        #batch_size = self.model.batch_size
        #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        #self.n_train_batches = n_train_batches
        l_t= T.cast(theano.tensor._shared(np.asarray(l_t,dtype=theano.config.floatX),borrow=True),'int32')
        l_v= T.cast(theano.tensor._shared(np.asarray(l_v,dtype=theano.config.floatX),borrow=True),'int32')
        self.layers = self.model.layers
        x = self.model.x
        y = self.model.y
        index = T.lscalar()  # index to a [mini]batch
        cost = self.model.cost
        params = self.model.params
        errors = self.model.errors
        train_model = theano.function(
                    inputs=[index],
                    outputs=[cost,errors],
                    updates=self.optimizer.get_updates(params=params,cost=cost),
                    givens={
                        x: train_set_x[l_t[index]:l_t[index+1]],
                        y: train_set_y[l_t[index]:l_t[index+1]]
                        }
                    )

        #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        #self.n_valid_batches = n_valid_batches
        validate_model = theano.function(
                    inputs=[index],
                    outputs=errors,
                    givens={
                        x: valid_set_x[l_v[index]:l_v[index+1]],
                        y: valid_set_y[l_v[index]:l_v[index+1]]
                        }
                    )

        return train_model, validate_model

    def train_by_sentence(self, x_train, y_train, x_val, y_val, index_train,index_val, save_model=False):
        """
        - train: {name:'sentenceID', data:[features], label:[labels]}
        """
        #train_model, validate_model = self._train_by_order_init_(train, val)
        train_model, validate_model = self._train_by_sentence_init_(x_train, y_train, x_val, y_val,index_train,index_val)
        patience = 10000
        patience_increase = 4
        improvement_threshold = 0.995
        validation_frequency = len(index_train)-1#min(self.n_train_batches, patience / 2)
        best_validation_loss = np.inf
        best_val_acc = 0.
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
        epoch = 0
        done_looping = False
        n_epochs = self.config['n_epochs']
        t_cost, t_acc, v_acc = [], [], []
        print 'start training...'
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            DropoutLayer.SetDropoutOn()
            for minibatch_index in xrange(len(index_train)-1):
                minibatch_avg_cost,train_acc = train_model(minibatch_index)
                iter = (epoch - 1) * (len(index_train)-1) + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    DropoutLayer.SetDropoutOff()
                    validation_losses = [validate_model(i) for i in xrange(len(index_val)-1)]
                    this_validation_loss = np.mean(validation_losses)
                    this_val_acc = 1 - this_validation_loss
                    this_train_acc = 1 - train_acc
                    print('epoch %i/%s, cost %.4f , train acc %.4f , val acc %.4f ' %(epoch,str(n_epochs),minibatch_avg_cost,(this_train_acc),(this_val_acc)))

                    t_cost.append(round(minibatch_avg_cost,5))
                    t_acc.append(round(this_train_acc,5))
                    v_acc.append(round(this_val_acc,5))


                    if save_model:
                        if this_val_acc > best_val_acc:
                            best_val_acc = this_val_acc
                            #print "best val acc at epoch %i is %.4f" %(epoch,best_val_acc)
                            folder = "./snapshot_{0}_{1}/".format(epoch, round(best_val_acc,3))
                            os.mkdir(folder)
                            tools.save_weights(self.layers, folder, epoch)
                            #print "model saved at epoch %i" %(epoch)

                    if this_validation_loss < best_validation_loss:
                        if (this_validation_loss < best_validation_loss * improvement_threshold):
                            patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter
        #            if this_train_acc - this_val_acc >0.05:
        #                done_looping = True
        #                break
        #        if patience <= iter:
        #            done_looping = True
        #            break
            if self.learning_rate_decay == True:
                if epoch % 5 == 0:
                    rate = theano.shared(np.cast[theano.config.floatX](0.5))
                    self.optimizer.lr = self.optimizer.lr * rate

        self.record = {
                      'training loss' : t_cost,
                      'training accuracy' : t_acc,
                      'validation accuracy' : v_acc }
        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% \n obtained at iteration %i, with test performance %f %%') %(best_validation_loss * 100., best_iter + 1, test_score * 100.))
    #print >> sys.stderr,('The code for file '+os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.))


    def _train_without_val_init_(self, train_set_x, train_set_y):
        self.learning_rate_decay = self.config['learning_rate_decay']
        batch_size = self.model.batch_size
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_train_batches = n_train_batches
        self.layers = self.model.layers
        x = self.model.x
        y = self.model.y
        index = T.lscalar()  # index to a [mini]batch
        cost = self.model.cost
        params = self.model.params
        errors = self.model.errors
        train_model = theano.function(
                    inputs=[index],
                    outputs=[cost,errors],
                    updates=self.optimizer.get_updates(params=params,cost=cost),
                    givens={
                        x: train_set_x[index * batch_size: (index + 1) * batch_size],
                        y: train_set_y[index * batch_size: (index + 1) * batch_size]
                        }
                    )
        return train_model

    def train_without_val(self, train_set_x, train_set_y, save_model=False):
        train_model = self._train_without_val_init_(train_set_x, train_set_y)
        patience = 10000
        patience_increase = 4
        improvement_threshold = 0.995
        validation_frequency = self.n_train_batches
        best_validation_loss = np.inf
        best_train_acc = 0.
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
        epoch = 0
        done_looping = False
        n_epochs = self.config['n_epochs']
        t_cost, t_acc, v_acc = [], [], []
        print 'start training...'
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            DropoutLayer.SetDropoutOn()
            for minibatch_index in xrange(self.n_train_batches):
                minibatch_avg_cost,train_acc = train_model(minibatch_index)
                this_train_acc = 1 - train_acc
            print('epoch %i/%s, cost %.4f , train acc %.4f ' %(epoch,str(n_epochs),minibatch_avg_cost,(this_train_acc)))

            if save_model:
                if this_train_acc > best_train_acc:
                    best_train_acc = this_train_acc
                    #print "best val acc at epoch %i is %.4f" %(epoch,best_val_acc)
                    folder = "./snapshot_{0}_{1}/".format(epoch, round(best_train_acc,3))
                    os.mkdir(folder)
                    tools.save_weights(self.layers, folder, epoch)
                    #print "model saved at epoch %i" %(epoch)

            if self.learning_rate_decay == True:
                if epoch % 5 == 0:
                    rate = theano.shared(np.cast[theano.config.floatX](0.5))
                    self.optimizer.lr = self.optimizer.lr * rate

        end_time = time.clock()

    def _train_init_(self, train_set_x, train_set_y, valid_set_x,valid_set_y):
        self.learning_rate_decay = self.config['learning_rate_decay']
        batch_size = self.model.batch_size
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_train_batches = n_train_batches
        self.layers = self.model.layers
        x = self.model.x
        y = self.model.y
        index = T.lscalar()  # index to a [mini]batch
        cost = self.model.cost
        params = self.model.params
        errors = self.model.errors
        train_model = theano.function(
                    inputs=[index],
                    outputs=[cost,errors],
                    updates=self.optimizer.get_updates(params=params,cost=cost),
                    givens={
                        x: train_set_x[index * batch_size: (index + 1) * batch_size],
                        y: train_set_y[index * batch_size: (index + 1) * batch_size]
                        }
                    )

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = n_valid_batches
        validate_model = theano.function(
                    inputs=[index],
                    outputs=errors,
                    givens={
                        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                        }
                    )

        return train_model, validate_model

    def train(self, train_set_x, train_set_y, valid_set_x, valid_set_y, save_model=False):
        train_model, validate_model = self._train_init_(train_set_x, train_set_y, valid_set_x, valid_set_y)
        patience = 10000
        patience_increase = 4
        improvement_threshold = 0.995
        validation_frequency = min(self.n_train_batches, patience / 2)
        best_validation_loss = np.inf
        best_val_acc = 0.
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
        epoch = 0
        done_looping = False
        n_epochs = self.config['n_epochs']
        t_cost, t_acc, v_acc = [], [], []
        print 'start training...'
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            DropoutLayer.SetDropoutOn()
            for minibatch_index in xrange(self.n_train_batches):
                minibatch_avg_cost,train_acc = train_model(minibatch_index)
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
                    DropoutLayer.SetDropoutOff()
                    validation_losses = [validate_model(i) for i in xrange(self.n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    this_val_acc = 1 - this_validation_loss
                    this_train_acc = 1 - train_acc
                    print('epoch %i/%s, cost %.4f , train acc %.4f , val acc %.4f ' %(epoch,str(n_epochs),minibatch_avg_cost,(this_train_acc),(this_val_acc)))

                    t_cost.append(round(minibatch_avg_cost,5))
                    t_acc.append(round(this_train_acc,5))
                    v_acc.append(round(this_val_acc,5))


                    if save_model:
                        if this_val_acc > best_val_acc:
                            best_val_acc = this_val_acc
                            #print "best val acc at epoch %i is %.4f" %(epoch,best_val_acc)
                            folder = "./snapshot_{0}_{1}/".format(epoch, round(best_val_acc,3))
                            os.mkdir(folder)
                            tools.save_weights(self.layers, folder, epoch)
                            #print "model saved at epoch %i" %(epoch)

                    if this_validation_loss < best_validation_loss:
                        if (this_validation_loss < best_validation_loss * improvement_threshold):
                            patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter
        #            if this_train_acc - this_val_acc >0.05:
        #                done_looping = True
        #                break
        #        if patience <= iter:
        #            done_looping = True
        #            break
            if self.learning_rate_decay == True:
                if epoch % 5 == 0:
                    rate = theano.shared(np.cast[theano.config.floatX](0.5))
                    self.optimizer.lr = self.optimizer.lr * rate

        self.record = {
                      'training loss' : t_cost,
                      'training accuracy' : t_acc,
                      'validation accuracy' : v_acc }
        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% \n obtained at iteration %i, with test performance %f %%') %(best_validation_loss * 100., best_iter + 1, test_score * 100.))
    #print >> sys.stderr,('The code for file '+os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.))
    def load(self):
        layers = self.model.layers
        dir = self.model.snapshot
        if not os.path.isdir(dir):
            raise IOError('no such snapshot file: %s' %(dir))

        snapshots = glob.glob(dir+'*.npy')
        e = self.config['e_snapshot']
        tools.load_weights(layers, dir, e)

    def predict_by_sentence(self, test_set_x, index_test, load_model=None, dropout=False):
        assert load_model != None, "load_model should be True of False"
        #batch_size = self.model.batch_size
        #n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        #self.n_test_batches = n_test_batches
        test_set_x = self.share_var(test_set_x,testing=True)
        layers = self.model.layers
        x = self.model.x
        y = self.model.y
        index = T.lscalar()  # index to a [mini]batch
        predict_times = len(index_test)-1
        index_test= T.cast(theano.tensor._shared(np.asarray(index_test,dtype=theano.config.floatX),borrow=True),'int32')
        if load_model == True:
            dir = self.model.snapshot
            if not os.path.isdir(dir):
                raise IOError('no such snapshot file: %s' %(dir))

            snapshots = glob.glob(dir+'*.npy')
            #e = os.path.basename(snapshots[0])[-5]
            e = self.config['e_snapshot']
            if dropout == False:
                tools.load_weights(layers, dir, e)
            else:
                tools.dropout_load_weights(layers, dir, e)

        test_model = theano.function(
                        inputs = [index],
                        outputs = self.model.y_pred,
                        givens={
                            x: test_set_x[index_test[index]:index_test[(index + 1)]],
                            }
                )

        n_test = test_set_x.get_value(borrow=True).shape[0]
        y_pred = np.array([])
        DropoutLayer.SetDropoutOff()
        print "predict on %d datas" %(int(n_test))
        for i in xrange(predict_times):
            y_pred = np.concatenate((y_pred,test_model(i)),axis=0)

        return y_pred

    def predict(self, test_set_x, load_model=None, dropout=False):
        assert load_model != None, "load_model should be True of False"
        #batch_size = self.model.batch_size
        #n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        #self.n_test_batches = n_test_batches
        layers = self.model.layers
        x = self.model.x
        y = self.model.y
        index = T.lscalar()  # index to a [mini]batch

        if load_model == True:
            dir = self.model.snapshot
            if not os.path.isdir(dir):
                raise IOError('no such snapshot file: %s' %(dir))

            snapshots = glob.glob(dir+'*.npy')
            #e = os.path.basename(snapshots[0])[-5]
            e = self.config['e_snapshot']
            if dropout == False:
                tools.load_weights(layers, dir, e)
            else:
                tools.dropout_load_weights(layers, dir, e)

        test_model = theano.function(
                        inputs = [index],
                        outputs = self.model.y_pred,
                        givens={
                            x: test_set_x[index:(index + 1)],
                            }
                )

        n_test = test_set_x.get_value(borrow=True).shape[0]
        y_pred = np.zeros(n_test)
        DropoutLayer.SetDropoutOff()
        print "predict on %d datas" %(int(n_test))
        for i in xrange(n_test):
            y_pred[i] = int(test_model(i))

        return y_pred

    def proba(self, X, load_model=None):
        assert load_model != None, "load_model should be True of False"
        layers = self.model.layers
        x = self.model.x
        y = self.model.y
        index = T.lscalar()  # index to a [mini]batch

        if load_model == True:
            dir = self.model.snapshot
            if not os.path.isdir(dir):
                raise IOError('no such snapshot file: %s' %(dir))

            snapshots = glob.glob(dir+'*.npy')
            #e = os.path.basename(snapshots[0])[-5]
            e = self.config['e_snapshot']
            tools.load_weights(layers, dir, e)

        prob_model = theano.function(
                        inputs = [index],
                        outputs = self.model.proba,
                        givens={
                            x: X[index:(index + 1)],
                            }
                )
        y_prob = []
        n_test = X.get_value(borrow=True).shape[0]
        DropoutLayer.SetDropoutOff()
        print "getting probability on %d datas" %(int(n_test))
        for i in xrange(n_test):
            y_prob.append(prob_model(i))
        return np.asarray(y_prob).reshape(n_test,y_prob[0].shape[1])
