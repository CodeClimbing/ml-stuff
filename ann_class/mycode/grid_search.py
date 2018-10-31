import sys
sys.path.append('../../ann_logistic_extra')

from process import get_data
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle



def _grid_search(hyperparameters):
	path = '/Users/rngentry/machinelearning/machine_learning_examples/ann_logistic_extra/ecommerce_data.csv'
	# Get and shuffle ecommerce data
	Xtrain, Ytrain, Xtest, Ytest = get_data(path)
	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Xtest, Ytest = shuffle(Xtest, Ytest)
	values = []
	keys = []
	for k, v in hyperparameters.items():
		keys.append(k)
		values.append(v)
	print(keys)
	print(values)
	combinations = list(itertools.product(*values))
	print(combinations)
	best_params = {}
	best_accuracy = 0
	for c in combinations:
		# Create the classifier model
		params = dict(zip(keys,c))
		print(params)
		model = MLPClassifier(**params)
		model.fit(Xtrain, Ytrain)
		# Train and test accuracy
		train_accuracy = model.score(Xtrain, Ytrain)
		test_accuracy = model.score(Xtest, Ytest)
		if test_accuracy > best_accuracy:
			best_accuracy = test_accuracy
			best_params = params
			print('Current best accuracy', best_accuracy)
			print('Current best params',params)
	print('The best accuaracy was', best_accuracy)
	print('With these parameters', best_params)


def main():

	
	hyperparameters = {'learning_rate_init': [1e-2,1e-4,1e-6], 'hidden_layer_sizes': [(5,5),(5,5,5),(10,10)], 'activation': ['relu','tanh'], 'max_iter': [500,1000,2000]}
	_grid_search(hyperparameters)

	'''
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

	'''




if __name__ == '__main__':
	main()