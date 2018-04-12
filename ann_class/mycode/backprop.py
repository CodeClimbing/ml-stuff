from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)

def sigmoid(Array):
	result = 1 / (1 - np.exp(-Array))
	return result

def softmax(Array):
	Array = Array - np.amax(Array)
	expA = np.exp(Array)
	result = expA / expA.sum(axis=1, keepdims=True)
	return result


def forward(X,W1,b1,W2,b2):
	Z1 = X.dot(W1) + b1
	A1 = sigmoid(Z1)
	Z2 = A1.dot(W2) + b2
	A2 = softmax(Z2)
	return A1, A2


def forward1(X,W1,b1,W2,b2):
	Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Z,Y


def derivative_wrt_w2(Y,Y_pred,A1):
	N, K = Y_pred.shape
	return A1.T.dot(Y - Y_pred)


def derivative_wrt_w1(X,W2,A1,Y,Y_pred):
	dJdA1 = (Y - Y_pred).dot(W2.T)
	result = X.T.dot(dJdA1 * A1 * (1 - A1))
	return result


def derivative_wrt_b2(Y,Y_pred):
	return (Y - Y_pred).sum(axis=0)


def derivative_wrt_b1(Y,Y_pred,W2,A1):
	return ((Y - Y_pred).dot(W2.T)* A1 * (1 - A1)).sum(axis=0)


def cost(Y, Y_pred):
	return (Y * np.log(Y_pred)).sum()


def classification_rate(Y, P):
   
	return np.mean(Y == P)
	"""
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total
	"""

def backprop(alpha,niter,Y,T,X,W1,b1,W2,b2):
	costs = []
	for i in range(niter):
		A1, Y_pred = forward1(X,W1,b1,W2,b2)
		if niter % 100 == 0:
			c = cost(T,Y_pred)
			P = np.argmax(Y_pred,axis=1)
			r = classification_rate(Y,P)
			print("cost:", c, "classification_rate:", r)
			costs.append(c)
		W2 += alpha * derivative_wrt_w2(T,Y_pred,A1)
		W1 += alpha * derivative_wrt_w1(X,W2,A1,T,Y_pred)
		b2 += alpha * derivative_wrt_b2(T,Y_pred)
		b1 += alpha * derivative_wrt_b1(T,Y_pred,W2,A1)
	return costs, W2, W1, b2, b1


def main():
	
	# create the data
	Nclass = 500
	D = 2 # dimensionality of input
	M = 3 # hidden layer size
	K = 3 # number of classes
	X1 = np.random.randn(Nclass, D) + np.array([0, -2])
	X2 = np.random.randn(Nclass, D) + np.array([2, 2])
	X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])
	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	N = len(Y)

	# turn Y into an indicator matrix for training
	T = np.zeros((N, K))
	for i in range(N):
		T[i, Y[i]] = 1

    # let's see what it looks like
	plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	plt.show()

    # randomly initialize weights
	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	alpha = 1e-3
	niters = 1000
	values = backprop(alpha,niters,Y,T,X,W1,b1,W2,b2)
	plt.plot(values[0])
	plt.show()

if __name__ == '__main__':
	main()


	