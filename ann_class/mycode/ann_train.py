from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from process import get_data
from sklearn.utils import shuffle

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


def sigmoid(Array):
	Array = Array - np.amax(Array)
	result = 1 / (1 - np.exp(-Array))
	return result


def forward(X,W1,b1,W2,b2):
	Z = np.tanh(X.dot(W1) + b1)
	return Z, softmax(Z.dot(W2) + b2)

def derivative_wrt_w2(Y,Y_pred,A1):
	N, K = Y_pred.shape
	return A1.T.dot(Y_pred - Y)


def derivative_wrt_w1(X,W2,A1,Y,Y_pred):
	dJdA1 = (Y_pred - Y).dot(W2.T) * (1 - A1*A1)
	result = X.T.dot(dJdA1)
	return result


def derivative_wrt_b2(Y,Y_pred):
	return (Y_pred - Y).sum(axis=0)


def derivative_wrt_b1(Y,Y_pred,W2,A1):
	dJdA1 = (Y_pred - Y).dot(W2.T) * (1 - A1*A1)
	return (dJdA1).sum(axis=0)


def cost(Y, Y_pred):
	return (Y * np.log(Y_pred)).sum()


def cross_entropy(Y,Y_pred):
	# add padding so log(0) error isn't thrown
	return -np.mean(Y * np.log(Y_pred))


def classification_rate(Y, P):
	return np.mean(Y == P)


def main():

	data_file = "/Users/rngentry/machinelearning/machine_learning_examples/ann_logistic_extra/ecommerce_data.csv"

	Xtrain, Ytrain, Xtest, Ytest = get_data(data_file)
	D = Xtrain.shape[1]
	K = len(set(Ytrain) | set(Ytest))
	M = 5
	print(Xtrain[0:4,:])

	Ytrain_ind = y2indicator(Ytrain, K)
	Ytest_ind = y2indicator(Ytest, K)

	
	W1 = np.random.randn(D,M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M,K)
	b2 = np.zeros(K)

	traincosts = []
	testcosts = []
	learningrate = 0.001

	for i in range(10000):
		A1, Ytrain_pred = forward(Xtrain,W1,b1,W2,b2)
		_ , Ytest_pred = forward(Xtest,W1,b1,W2,b2)
		ctrain = cross_entropy(Ytrain_ind,Ytrain_pred)
		ctest = cross_entropy(Ytest_ind,Ytest_pred)
		traincosts.append(ctrain)
		testcosts.append(ctest)

		if i % 1000 == 0:
			print(i,ctrain,ctest)

		W2 -= learningrate * derivative_wrt_w2(Ytrain_ind,Ytrain_pred,A1)
		b2 -= learningrate * derivative_wrt_b2(Ytrain_ind,Ytrain_pred)
		W1 -= learningrate * derivative_wrt_w1(Xtrain,W2,A1,Ytrain_ind,Ytrain_pred)
		b1 -= learningrate * derivative_wrt_b1(Ytrain_ind,Ytrain_pred,W2,A1)



	# classification rate
	Ptrain = np.argmax(Ytrain_pred,axis=1)
	Ptest = np.argmax(Ytest_pred,axis=1)
	crtrain = classification_rate(Ytrain,Ptrain)
	crtest = classification_rate(Ytest,Ptest)

	print("Final train classification_rate:", crtrain)
	print("Final test classification_rate:", crtest)

	legend1, = plt.plot(traincosts, label='train cost')
	legend2, = plt.plot(testcosts, label='test cost')
	plt.legend([legend1, legend2])
	plt.show()




if __name__ == '__main__':
	main()