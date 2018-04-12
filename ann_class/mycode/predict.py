import numpy as np
from process import get_data
import sys




def tanh(Array):
	expA = np.exp(Array)
	neg_expA = np.exp(-Array)
	result = (expA - neg_expA) / (expA + neg_expA)
	return result

def softmax(Array):
	expA = np.exp(Array)
	result = expA / expA.sum(axis=1, keepdims=True)
	return result


def forward(X, W1, b1, W2, b2):
	Z1 = tanh(X.dot(W1) + b1)
	Z2 = softmax(X.dot(W2) + b2)
	return Z2

def classification_rate(Y_pred, Y):
	assert Y_pred.shape == Y.shape
	return np.mean(Y_pred == Y)




def main():
	filepath = sys.argv[1]
	X, Y, _, _ = get_data(filepath)

	# randomly initialize weights
	M = 5
	D = X.shape[1]
	K = len(set(Y))
	W1 = np.random.randn(D, M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K)
	b2 = np.zeros(K)
	
	# predicted Y
	Y_pred = forward(X,W1,b1,W2,b2)

	print(Y_pred[:10,:])
	y_pred = np.argmax(Y_pred, axis=1)
	# get classification rate
	correct = classification_rate(y_pred, Y)
	print('The classification rate is:', correct)



if __name__ == '__main__':
	main()