from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from process import get_data

def y2indicator(y,K):
	N = len(y)
	Y_ind = np.zeros((N,K))
	for i in range(N):
		Y_ind[i, y[i]] = 1
	return Y_ind


def softmax(Array):
	Array = Array - np.amax(Array)
	expA = np.exp(Array)
	return expA / expA.sum(axis=1,keepdims=True)


def forward(X,W,b):
	return softmax(X.dot(W)+b)


def predict(Y_pred):
	return np.argmax(Y_pred, axis=1)


def classification_rate(Y,Y_pred):
	return np.mean(Y == Y_pred)


def cross_entropy(Y,Y_pred):
	return -np.mean(Y * np.log(Y_pred))


def main():

	data_file = "/Users/rngentry/machinelearning/machine_learning_examples/ann_logistic_extra/ecommerce_data.csv"
	# Get data
	Xtrain, Ytrain, Xtest, Ytest = get_data(data_file)
	D = Xtrain.shape[1]
	K = len(set(Ytrain) | set(Ytest))
	
	# convert to indicator matrices
	Ytrain_ind = y2indicator(Ytrain, K)
	Ytest_ind = y2indicator(Ytest, K)

	# initialize weights and biases
	W = np.random.randn(D,K)
	b = np.zeros(K)


	traincosts = []
	testcosts = []
	alpha = 1e-3

	for i in range(10000):
		# calculate predicted Y given X
		Ytrain_pred = forward(Xtrain,W,b)
		Ytest_pred = forward(Xtest,W,b)

		# Calculate cost and classification rate
		cr_train = classification_rate(Ytrain_ind,Ytrain_pred)
		cr_test = classification_rate(Ytest_ind,Ytest_pred)

		traincost = cross_entropy(Ytrain_ind,Ytrain_pred)
		testcost = cross_entropy(Ytest_ind,Ytest_pred)
		
		traincosts.append(traincost)
		testcosts.append(testcost)

		W -= alpha * Xtrain.T.dot(Ytrain_pred - Ytrain_ind)
		b -= alpha * (Ytrain_pred - Ytrain_ind).sum(axis=0)
		
		if i % 1000 == 0:
			print(i,traincost,testcost)

	print("Final train classification_rate:", classification_rate(Ytrain, predict(Ytrain_pred)))
	print("Final test classification_rate:", classification_rate(Ytest, predict(Ytest_pred)))
	legend1, = plt.plot(traincosts, label='train cost')
	legend2, = plt.plot(testcosts, label='test cost')
	plt.legend([legend1, legend2])
	plt.show()

if __name__ == '__main__':
	main()
