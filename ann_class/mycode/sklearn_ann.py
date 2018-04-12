import sys
sys.path.append('../../ann_logistic_extra')

from process import get_data

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

def main():

	path = '/Users/rngentry/machinelearning/machine_learning_examples/ann_logistic_extra/ecommerce_data.csv'
	# Get and shuffle ecommerce data
	Xtrain, Ytrain, Xtest, Ytest = get_data(path)
	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Xtest, Ytest = shuffle(Xtest, Ytest)
	
	# Create the classifier model
	model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000)

	# Train the model
	model.fit(Xtrain, Ytrain)

	# Train and test accuracy
	train_accuracy = model.score(Xtrain, Ytrain)
	test_accuracy = model.score(Xtest, Ytest)
	print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)






if __name__ == '__main__':
	main()