from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, softmax, cost, to_indicator_matrix, get_train_test, error_rate, relu, init_weight_bias
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


class ANN(object):

	def __init__(self, hidden_layer_sizes=(10,10), activation='relu', learning_rate_init=1e-7, max_iter=1000, batch_size='auto'):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.num_layers = len(hidden_layer_sizes) + 2
		self.activation = activation
		self.learning_rate_init = learning_rate_init
		self.max_iter = max_iter
		self.batch_size = batch_size


	def fit(self,X,y,plot_cost=False):
		X_train, Y_train, X_test, Y_test = get_train_test(X,y,percent_train=0.7)
		n, d = X_train.shape
		k = Y_train.shape[1]

		self.W1, self.b1 = init_weight_bias(d,self.hidden_layer_sizes[0])
		self.W2, self.b2 = init_weight_bias(self.hidden_layer_sizes[0],k)
		costs = []
		best_validation_error = 1

		if (self.batch_size == 'auto'):
			self.batch_size = min(200,n)

		num_batches = int(n / self.batch_size)

		
		for i in range(self.max_iter):
			X_temp, Y_temp = shuffle(X_train,Y_train)
			for j in range(num_batches):
				X_temp, Y_temp = X_train[j * self.batch_size:j * self.batch_size + self.batch_size,:], Y_train[j * self.batch_size:j * self.batch_size + self.batch_size,:]
				Ypred, Z1 = self.forward(X_temp) 
			
				pY_t = Ypred - Y_temp
				self.W2 -= self.learning_rate_init * (Z1.T.dot(pY_t))
				self.b2 -= self.learning_rate_init * (pY_t.sum(axis=0))
				dZ = pY_t.dot(self.W2.T) * (Z1 > 0)
				self.W1 -= self.learning_rate_init * X_temp.T.dot(dZ)
				self.b1 -= self.learning_rate_init * dZ.sum(axis=0)

			if (i % 2) == 0:
				pY_test, _ = self.forward(X_test)
				c = cost(Y_test,pY_test)
				costs.append(c)
				e = error_rate(Y_test.argmax(axis=1),pY_test.argmax(axis=1))
				print('Iteration', i, 'Cost:', c, 'Error Rate:', e)
				if e < best_validation_error:
					best_validation_error = e
		print("best_validation_error:", best_validation_error)
		
		if plot_cost:
			plt.plot(costs)
			plt.show()


	def _backprop(self, X, y):
		pass



	def _forward_pass(self, X):
		
		# List of activations at each layer. 1st layer is the inputs
		activation_values = []
		activation_values.append(X)
		
		# Hidden layer activation function
		hidden_activation = ACTIVATIONS[self.activation]
		
		# Do the forward pass and save the activations at each layer
		for i in range(self.num_layers - 1):
			weight, bias = weight_biases[i]
			activation_values[i+1] = (activation_values[i].dot(weight)) + bias
			if (i + 1) != (self.num_layers - 1):
				activation_values[i+1] = hidden_activation(activation_values[i+1])

		# Output Activation
		output_activation = ACTIVATIONS[self.output_activation]
		activation_values[i+1] = output_activation[activation_values[i+1]]
		
		return activation_values


	def forward(self,X):
		Z = relu(X.dot(self.W1) + self.b1)
		Ypred = softmax(Z.dot(self.W2) + self.b2)
		return Ypred, Z




	def predict(self,X):
		Ypred, _ = self.forward(X)
		return np.argmax(Ypred,axis=1)


	def score(self,X,y):
		ypred = self.predict(X)
		return np.mean(y == ypred)


def main():
	# Uses sklearn
	
	X, y = get_data()
	X, y = shuffle(X, y)
	Y = to_indicator_matrix(y)
	hidden_layer_sizes = np.array([200])
	Xtrain, Ytrain = X[:-1000,:], Y[:-1000,:]
	Xtest, Ytest = X[-1000:,:], Y[-1000:,:]
	# Create the classifier model
	model = MLPClassifier(hidden_layer_sizes, learning_rate_init=1e-8, max_iter=1000, activation='relu', verbose=True)

	# Train the model
	model.fit(Xtrain, Ytrain)

	# Train and test accuracy
	train_accuracy = model.score(Xtrain, Ytrain)
	test_accuracy = model.score(Xtest, Ytest)
	print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)
	"""

	# Uses ANN class
	
	X, y = get_data()
	hidden_layer_sizes = np.array([200])
	model = ANN(hidden_layer_sizes,max_iter=100,learning_rate_init=.001)
	model.fit(X,y,plot_cost=True)
	print(model.score(X,y))	
	"""

if __name__ == '__main__':
	main()




