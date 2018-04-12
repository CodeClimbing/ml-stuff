import numpy as np
import nonlinearity as nl
import matplotlib.pyplot as plt






def create_indicator(Array):
	indicator = Array.argmax(axis=1)
	result = np.eye(Array.shape[1])[indicator]
	return result


def forward(X, W1, b, W2, b2):

	Z1 = nl.sigmoid(X.dot(W1) + b)
	Z2 = nl.softmax(Z1.dot(W2) + b2)
	#Y = create_indicator(Z2)
	return Z2

def classification_rate(Y,P):
	assert len(Y) == len(P)
	correct = np.sum(Y == P)
	total = len(Y)
	return correct / total

def main():
	samples = 500
	X1 = np.random.randn(samples, 2) + np.array([0, -2])
	X2 = np.random.randn(samples, 2) + np.array([2, 2])
	X3 = np.random.randn(samples, 2) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])
	Y = np.array([0]*samples + [1]*samples + [2]*samples)
	plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	plt.show()
	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes
	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	P_Y_given_X = forward(X, W1, b1, W2, b2)
	print(P_Y_given_X.shape)
	P = np.argmax(P_Y_given_X, axis=1)
	print(P.shape)
	print("Classification rate for randomly chosen weights:", classification_rate(Y, P))

if __name__ == '__main__':
	main()