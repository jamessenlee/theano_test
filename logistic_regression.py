from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import matplotlib.pyplot as plt
import types

theano.config.compute_test_value = 'off'

mini_datasets = "mini_data_sets"


def gen_mini_datasets(data_xy,size):
    dx,dy = data_xy
    mini_dx = dx[:size,:]
    mini_dy = dy[:size] 

    return [dx,dy]


def check_datasets_shape(data_xy):
    dx,dy = data_xy

    print (dx.shape)
    print (dy.shape)



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load mini dataset
    if False: #os.path.isfile(mini_datasets):
	print ("read mini datasets")
#        with open(mini_datasets) as f:
#            train_set, valid_set, test_set = pickle.load(f)
 
#	print("checking train set shape")
#	check_datasets_shape(train_set)	

#	print("checking valid set shape")
#	check_datasets_shape(valid_set)	

#	print("checking test set shape")
#	check_datasets_shape(test_set)	
#	os._exit(1)
    else:
	#load org dataset
    	with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

 #           print ('dumping mini datasets')
	    
#	    mini_train = gen_mini_datasets(train_set,3)	
#	    mini_valid = gen_mini_datasets(valid_set,3)	
#	    mini_test  = gen_mini_datasets(test_set,3)	
#	    md = (mini_train,mini_valid,mini_test)
#	    print (type(md))
#	    with open(mini_datasets, 'wb') as f:
#        	pickle.dump(md,f)
  
#	    os._exit(1)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')



    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y),(test_set),(valid_set),(train_set)]
    return rval


learning_rate = 0.001
input_size  = 28*28
output_size = 10
batch_size = 600
n_epochs = 1000


datasets = load_data('mnist.pkl.gz')
train_set_x,train_set_y = datasets[0]
valid_set_x,valid_set_y = datasets[1]
test_set_x,test_set_y = datasets[2]


class  LogisticRegression(object):
	def __init__(self,inputs,outputs):
	    self.W = theano.shared(np.zeros((input_size,output_size),dtype=np.float64),name="W",borrow=True)	
	    self.b = theano.shared(np.zeros(output_size,dtype=np.float64),name="b",borrow=True)	
	    self.param = [self.W,self.b]
		
	    #batch * input_size
	    self.inputs = inputs
	    #batch * output_size
	    self.outputs = outputs
		

	    #batch * output_size
	    self.p_y_given_x = T.nnet.softmax(T.dot(self.inputs,self.W) + self.b)	
	    self.pred = T.argmax(self.p_y_given_x,axis=1)
	    self.cost = self.negative_log_likelihood(self.p_y_given_x,self.outputs)

	    self.g_W = T.grad(self.cost,self.W)
	    self.g_b = T.grad(self.cost,self.b)

	    self.updates = [(self.W, self.W - learning_rate * self.g_W),(self.b,self.b - learning_rate * self.g_b)]
		

	# negative Log likelihood
	def negative_log_likelihood(self,p_y_given_x,y):
#		p_y_given_x = theano.printing.Print('p_y_given_x')(p_y_given_x)
#		y = theano.printing.Print('y ')(y)
		
#		log_p_y_given_x = T.log(p_y_given_x)

#		log_p_y_given_x = theano.printing.Print('log_p')(log_p_y_given_x)

#		llh_val = log_p_y_given_x[T.arrange(y.shape[0]),y]
#		llh_val = theano.printing.Print('llh_val')(llh_val)

#		return -T.mean(llh_val)  
		return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
	
	def error(self,):
		return T.mean(T.neq(self.pred, self.outputs))
		
	
	def predict(self,):
	

		return None


def mnist_train(data_sets):
#	train_set_x,train_set_y = datasets[0]
#	valid_set_x,valid_set_y = datasets[1]
#	test_set_x,test_set_y = datasets[2]


	#plot the first img
	#otx,oty = datasets[5]
	#plt.figure()
	#img_0 = otx[0].reshape(28,28)
	#plt.imshow(img_0)
	#plt.show()

	x = T.matrix('x')
	y = T.ivector('y')
	
	classifier = LogisticRegression(x,y)
	

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size

	index = T.lscalar()

	test_model = theano.function(
		inputs =[index],
		outputs = classifier.error(),
		givens = {
			x:test_set_x[index*batch_size:(index+1)*batch_size],
			y:test_set_y[index*batch_size:(index+1)*batch_size]
			}
		
		)

	valid_model = theano.function(
		inputs =[index],
		outputs = classifier.error(),
		givens = {
			x:valid_set_x[index*batch_size:(index+1)*batch_size],
			y:valid_set_y[index*batch_size:(index+1)*batch_size]
			}
		
		)

	train_model = theano.function(
		inputs =[index],
		outputs = classifier.cost,
		updates = classifier.updates,
		givens = {
			x:train_set_x[index*batch_size:(index+1)*batch_size],
			y:train_set_y[index*batch_size:(index+1)*batch_size]
			}
		
		)


	#training process
	print("training model")
	
	epoch = 0
	
	patience = 5000
	patience_increase = 2
	improvement_threshold = 0.995
	validation_freq = min(n_train_batches,patience//2)
	best_valid_loss = np.inf
	test_score = 0
	start_time = timeit.default_timer()

	done_looping = False
	valid_avg_loss = np.inf
	test_avg_loss = np.inf

	while(epoch < n_epochs) and (not done_looping):

	    epoch += 1

	    for minibatch_index in range(n_train_batches):
#		print ("epoch:%d,batch_index:%d"% (epoch,minibatch_index))
		train_mini_batch = train_model(minibatch_index)
	
		iter = (epoch)*n_train_batches + minibatch_index

		if (iter +1) % validation_freq == 0:
		    valid_losses = [valid_model(i) 
			for(i) in range(n_valid_batches)]
		    valid_avg_loss = np.mean(valid_losses)
		
		    print('epoch %i,minibatch %i/%i,validation error %f %%' %
		    (
			epoch,minibatch_index+1,n_train_batches,valid_avg_loss * 100.
		    ))

		    if valid_avg_loss < best_valid_loss:
			if(valid_avg_loss < best_valid_loss*improvement_threshold):
				patience = max(patience,iter*patience_increase)

			best_valid_loss = valid_avg_loss

			test_losses = [test_model(i) 
				for i in range(n_test_batches)]
			test_avg_loss = np.mean(test_losses)
	
 			print(('epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%') %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_avg_loss * 100.
                            ))

                        # save the best model
                        with open('best_model.pkl', 'wb') as f:

                            pickle.dump(classifier, f)

		if(patience <= iter):
			done_loop = True
			break
	
	end_time = timeit.default_timer()
    	print(
        	(
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        	)
        	% (best_valid_loss * 100., test_avg_loss * 100.)

        	)

    	print('The code run for %d epochs, with %f epochs/sec' % (

        	epoch, 1. * epoch / (end_time - start_time)))

    	print(('The code for file ' +

           os.path.split(__file__)[1] +

           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

if __name__ == '__main__':
#	datasets = load_data('mnist.pkl.gz')
	mnist_train(datasets)

