from __future__ import print_function, division
from builtins import range
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from utils import get_data, softmax, cost, to_indicator_matrix, get_train_test, error_rate, relu, init_weight_bias
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


class BaseANN(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self, hidden_layer_sizes, activation, learning_rate_init, max_iter, batch_size, loss):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.num_layers = len(hidden_layer_sizes) + 2
		self.activation = activation
		self.learning_rate_init = learning_rate_init
		self.max_iter = max_iter
		self.batch_size = batch_size
		self.loss = loss



	# Initializes weights and biases attributes
	def _init_weights_biases(X, Y):
	self.weights_ = []
	self.biases_ = []
	layer_sizes = [self.num_features_] + list(hidden_layers) + [self.num_classes_]
	for i in range(self.num_layers_ - 1):
		W, b = init_weight_bias(layer_sizes[i],layer_sizes[i+1])
		self.weights.append(W)
		self.biases.append(b)
	

	def _pack_params(self, params):
		pass


	def _forward_pass(self, X):
		
		# List of activations at each layer. 1st layer is the inputs
		activation_values = [X]
		
		# Hidden layer activation function
		hidden_activation = ACTIVATIONS[self.activation]
		
		# Do the forward pass and save the activations at each layer
		for i in range(self.num_layers - 1):
			weight, bias = self.weight_biases[i]
			activation_values[i+1] = (activation_values[i].dot(weight)) + bias
			if (i + 1) != (self.num_layers - 1):
				activation_values[i+1] = hidden_activation(activation_values[i+1])

		# Output Activation
		output_activation = ACTIVATIONS[self.output_activation]
		activation_values[i+1] = output_activation[activation_values[i+1]]
		
		return activation_values


	'''
	Computes the gradients for the weights and biases at layer
	Returns
	-------
	coef_grads: A list of the gradients for the weights at each layer
	intercept_grads: A list of the gradients for the intercepts at each layer
	'''
	def _compute_loss_grads(layer,n_samples,activations,deltas,coef_grads,intercept_grads):
		coef_grads[layer] = activations[layer].T.dot(deltas[layer])
		coef_grads[layer] += self.alpha * self.weights[layer]
		coef_grads[layer] /= n_samples
		intercept_grads[layer] = np.mean(deltas[layer], axis=0)
		return coef_grads, intercept_grads


	def _backprop(self, X, y activation_values, deltas, coef_grads,
                  intercept_grads):
		
		n_samples = X.shape[0]

		# Do forward prop to get activation values
		activation_values = self._forward_pass(X)

		# Get loss function
		loss_function = self.loss
		if loss_function == 'log_loss' and self.out_activation_ == 'logistic':
            loss_function = 'binary_log_loss'
        
        # Compute loss
        loss = LOSS_FUNCTIONS[loss_function](y, activation_values[-1])
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self.weights]))
        loss += self.alpha * values / n_samples


        # Compute the deltas at each layer and using those compute the gradients
		derivative_function = DERIVATIVES[self.activation]
		last = self.num_layers - 2
		deltas[last] = activation_values[-1] - y
		for i in range(last,0,-1):
			deltas[i - 1] = deltas[i].dot(self.weights[i].T)
			derivative_function(activation_values[i],deltas[i-1])
			coef_grads, intercept_grads = _compute_loss_grads(i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

		return loss, coef_grads, intercept_grads


	def _predict(self, X):
		activations = self._forward_pass(X)
		y_pred = activations[-1]
		return y_pred


	def _score(self, X, y):
		y_pred = self._predict(X)
		return np.mean(y == y_pred)

	'''
	Implements grid search for hyperparameters
	Params
	------
	X: validation set of inputs
	y: validation set of outputs
	hyperparameters: Dictionary with a key for each hyperparameter and a list of values
					 e.g. {learning_rate: [.01, .001, .00001], alpha: [.001,.00001,.000001]}
	
	Returns
	------
	best_hyperparameters: The values of the hyperparameters that give the lowest cost
	'''

	def _grid_search(self,X,y,hyperparameters):
		self._init_attributes()
		values = []
		keys = []
		for k, v in hyperparameters:
			keys.append(k)
			values.append(v)
		combinations = list(itertools.product(*v))
		params_dict = {}
		for c in combinations:



