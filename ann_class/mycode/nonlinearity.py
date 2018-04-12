import numpy as np

def softmax(Array):
	expA = np.exp(Array)
	result_array = expA / expA.sum(axis=1, keepdims=True)
	return result_array


def sigmoid(Array):
	result_array = 1 / (1 + np.exp(-Array))
	return result_array

def tanh(Array):
	expA = np.exp(Array)
	neg_expA = np.exp(-Array)
	result_array = (expA - neg_expA) / (expA + neg_expA)



def main():
	A = np.arange(9.0).reshape((3,3))
	print(A)
	print(softmax(A))


if __name__ == '__main__':
	main()
