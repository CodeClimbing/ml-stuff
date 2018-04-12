import numpy as np 
import pandas as pd
import sys
import os






def get_data(filepath):
	df = pd.read_csv(filepath)

	data = df.as_matrix()

	np.random.shuffle(data)

	X = data[:,:-1]
	Y = data[:,-1].astype(np.int32)

	N, D = X.shape
	X2 = np.zeros((N, D+3))
	X2[:,0:(D-1)] = X[:,0:(D-1)]
	
	Z = np.zeros((N,4))
	Z[np.arange(N),X[:,D-1].astype(np.int32)] = 1
	X2[:,-4:] = Z

	X = X2
	# One hot encode the class variable
	#Z = np.eye(4)[X[:,-1].astype(np.int32)]

	# Combine X and Z
	#X = np.concatenate((X[:,-1].reshape(X.shape[0],1),Z), axis=1)

	# Split into train and test sets
	Xtrain = X[:-100]
	Ytrain = Y[:-100]
	Xtest = X[-100:]
	Ytest = Y[-100:]

	# normalize columns 1 and 2
	for i in (1, 2):
		m = Xtrain[:,i].mean()
		s = Xtrain[:,i].std()
		Xtrain[:,i] = (Xtrain[:,i] - m) / s
		Xtest[:,i] = (Xtest[:,i] - m) / s

	return Xtrain, Ytrain, Xtest, Ytest


def get_binary_data():
  	# return only the data from the first 2 classes
 	Xtrain, Ytrain, Xtest, Ytest = get_data()
 	X2train = Xtrain[Ytrain <= 1]
 	Y2train = Ytrain[Ytrain <= 1]
 	X2test = Xtest[Ytest <= 1]
 	Y2test = Ytest[Ytest <= 1]
 	return X2train, Y2train, X2test, Y2test





def main():
	filepath = sys.argv[1]
	print(filepath)
	get_data(filepath)




if __name__ == '__main__':
	main()