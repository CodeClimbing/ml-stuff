from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

from utils import get_data, softmax, cost, to_indicator_matrix, get_train_test, error_rate, relu, init_weight_bias
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


class ANN(object):

	def __init__(self, hidden_layer_sizes=(1,), activation='relu', learning_rate_init=1e-7, max_iter=1000):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation = activation
		self.learning_rate_init = learning_rate_init
		self.max_iter = max_iter


	def fit(self,X,y,plot_cost=False):
		X, y = shuffle(X, y)
		n, d = X.shape
		Y = to_indicator_matrix(y)
		k = Y.shape[1]
		split = int(0.7 * n)
		yvalid = y[split:]
		Xvalid, Yvalid = X[split:,:], Y[split:,:]
		X, Y = X[:split,:], Y[:split,:]
		X, Y, Xvalid, Yvalid = get_train_test(X,y,percent_train=0.7)


		self.W1, self.b1 = init_weight_bias(d,self.hidden_layer_sizes[0])
		self.W2, self.b2 = init_weight_bias(self.hidden_layer_sizes[0],k)
		costs = []
		best_validation_error = 1
		
		for i in range(self.max_iter):
			Ypred, Z1 = self.forward(X) 
			
			pY_t = Ypred - Y
			self.W2 -= self.learning_rate_init * (Z1.T.dot(pY_t))
			self.b2 -= self.learning_rate_init * (pY_t.sum(axis=0))
			dZ = pY_t.dot(self.W2.T) * (Z1 > 0)
			self.W1 -= self.learning_rate_init * X.T.dot(dZ)
			self.b1 -= self.learning_rate_init * dZ.sum(axis=0)

			if (i % 10) == 0:
				pYvalid, _ = self.forward(Xvalid)
				c = cost(Yvalid,pYvalid)
				costs.append(c)
				e = error_rate(yvalid,pYvalid.argmax(axis=1))
				print('Iteration', i, 'Cost:', c, 'Error Rate:', e)
				if e < best_validation_error:
					best_validation_error = e
		print("best_validation_error:", best_validation_error)
		
		if plot_cost:
			plt.plot(costs)
			plt.show()



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
	"""
	X, y = get_data()
	X, y = shuffle(X, y)
	Y = to_indicator_matrix(y)
	hidden_layer_sizes = np.array([200])
	Xtrain, Ytrain = X[:-1000,:], Y[:-1000,:]
	Xtest, Ytest = X[-1000:,:], Y[-1000:,:]
	# Create the classifier model
	model = MLPClassifier(hidden_layer_sizes, learning_rate_init=1e-7, max_iter=1000, activation='relu', verbose=True)

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
	model = ANN(hidden_layer_sizes)
	model.fit(X,y,plot_cost=True)
	print(model.score(X,y))	
	

if __name__ == '__main__':
	main()




