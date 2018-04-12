from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd
import os


dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

data_file = '/fer2013/fer2013.csv'




def init_weight_bias(in_size, out_size):
	W = np.random.randn(in_size,out_size) / np.sqrt(in_size)
	b = np.zeros(out_size)
	return W.astype(float), b.astype(float)


def init_weights_biases(X, Y, hidden_layers):
	input_size = X.shape[1]
	if len(Y.shape) == 1:
		output_size = 1
	else:
		output_size = Y.shape[1]
	weights = []
	biases = []
	layer_sizes = [input_size] + list(hidden_layers) + [output_size]
	for i in range(len(layer_sizes) - 1):
		W, b = init_weight_bias(layer_sizes[i],layer_sizes[i+1])
		weights.append(W)
		biases.append(b)
	return weights, biases


def create_pixel_matrix(pixel_data):
	num_column = len(pixel_data[0].split())
	num_row = len(pixel_data)
	X = np.empty((num_row,num_column))
	for i in range(num_row):
		X[i] = pixel_data[i].split()
	return X


def balance_class(X,Y):
	X0, Y0 = X[Y != 1,:], Y[Y != 1]
	X1 = X[Y == 1,:]
	X1 = np.repeat(X1,9,axis=0)
	X = np.vstack((X0,X1))
	Y = np.concatenate((Y0,[1] * len(X1)))
	return X, Y


def get_data(balance=True):

	df = pd.read_csv(dir_path + data_file)
	data = df.as_matrix()


	Y = data[:,0].astype(int)
	pixel_data = data[:,1]
	
	#ind = data[:,2]

	X = create_pixel_matrix(pixel_data)


	if balance:
		balance_class(X,Y)
	"""
	Xtrain, Ytrain = X[ind == 'Training'], Y[ind == 'Training']
	Xpritest, Ypritest = X[ind == 'PrivateTest'], Y[ind == 'PrivateTest']
	Xpubtest, Ypubtest = X[ind == 'PublicTest'], Y[ind == 'PublicTest']
	"""


	return X / X.max(), Y


def get_image_data():
	X, Y = get_data()
	N, D = X.shape
	d = int(np.sqrt(D))
	X = X.reshape(N,1,d,d)
	print(X.shape)
	return X, Y


def get_binary_data(balance=True):
	df = pd.read_csv(dir_path + data_file)
	data = df.as_matrix()
	Y0 = data[:,0].astype(float)
	Y = Y0[Y0 <= 1]
	pixel_data = data[Y0 <= 1,1]
	X = create_pixel_matrix(pixel_data)

	if balance:
		X, Y = balance_class(X, Y)

	return X / X.max(), Y


def relu(Array):
	result = Array * (Array > 0)
	return result

def tanh(Array):
	expA = np.exp(Array)
	neg_expA = np.exp(-Array)
	result = (expA - neg_expA) / (expA + neg_expA)


def sigmoid(Array):
	result = 1 / (1 + np.exp(-Array))
	return result


def softmax(Array):
	Array = Array - np.amax(Array)
	expA = np.exp(Array)
	result = expA / expA.sum(axis=1,keepdims=True)
	return result

ACTIVATIONS = {'relu': relu, 'tanh': tanh, 'logistic': sigmoid, 'softmax': softmax}


def to_indicator_matrix(array):
	num_rows = len(array)
	num_cols = len(set(array))
	M = np.zeros((num_rows,num_cols))
	for i in range(num_rows):
		M[i, array[i]] = 1
	return M


# input: X = array of training examples, features; y = vector of class 
# output: Xtrain, Xtest, Ytrain, Ytest where Y is an indicator matrix
def get_train_test(X,y,percent_train,shuffle=True):
	if shuffle:
		X, y = shuffle(X, y)
	n, d = X.shape
	Y = to_indicator_matrix(y)
	k = Y.shape[1]
	split = int(percent_train * n)
	Xtest, Ytest = X[split:,:], Y[split:,:]
	Xtrain, Ytrain = X[:split,:], Y[:split,:]
	return Xtrain, Ytrain, Xtest, Ytest


def error_rate(y,ypred):
	return np.mean(y != ypred)


def cost(Y,Ypred):
  	return -(Y * np.log(Ypred)).sum()


def main():
	X,Y = get_data()
	print(X.shape)
	hidden_layers = (10,12,13)
	weights, biases = init_weights_biases(X,Y,hidden_layers)
	print(len(weights),weights[0].shape,weights[1].shape)
	print(len(biases))


if __name__ == '__main__':
	main()


