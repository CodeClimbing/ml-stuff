from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
	




# tensor flow variables are not the same as regular Python variables
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2



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

	tfX = tf.placeholder(tf.float32, [None, D])
	tfY = tf.placeholder(tf.float32, [None, K])

	predict_model = tf.layers.Dense(units=M,activation=tf.nn.tanh,use_bias=True)
	Ypred = predict_model(tfX)


	"""
	W1 = init_weights([D, M]) # create symbolic variables
	b1 = init_weights([M])
	W2 = init_weights([M, K])
	b2 = init_weights([K])
	"""

	costf = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ypred, labels=tfY)


	"""
	logits = forward(tfX, W1, b1, W2, b2)

	cost = tf.reduce_mean(
  	tf.nn.softmax_cross_entropy_with_logits(
    labels=tfY,
    logits=logits
  	)
	) # compute costs
	"""
	train_op = tf.train.GradientDescentOptimizer(0.05).minimize(costf) # construct an optimizer
	# input parameter is the learning rate

	predict_op = tf.argmax(Ypred, 1)
	# input parameter is the axis on which to choose the max

	writer = tf.summary.FileWriter('.')
	writer.add_graph(tf.get_default_graph())
	# just stuff that has to be done
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)


	for i in range(1000):
		sess.run(train_op, feed_dict={tfX: X, tfY: T})
		pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
		if i % 100 == 0:
			print("Accuracy:", np.mean(Y == pred))
	print(sess.run(Ypred, feed_dict={tfX: X}))

if __name__ == '__main__':
	main()