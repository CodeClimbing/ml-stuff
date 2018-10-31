
import numpy as np


class BaseOptimizer(metaclass=ABCMeta):

	def __init__(self, params, learning_rate_init=0.1):
		self.params = [param for param in params]
		self.learning_rate_init = learning_rate_init
		self.learning_rate = float(learning_rate_init)

	def update_params(self, grads):
		updates = _get_updates(grads)
		self.params = [param + update for param, update in zip(self.params,updates)]


class StochasticOptimizer(BaseOptimizer):

	def __init__(self, params, learning_rate_init=0.1, momentum=0.9, nesterov=True):
		super(StochasticOptimizer, self).__init__(params, learning_rate_init)

		self.momentum = momentum
		self.nesterov = nesterov
		self.velocities = [np.zeros_like(param) for param in params]


	def _get_updates(self, grads):

		self.velocities = [(self.momentum * velocity - self.learning_rate * grad) for velocity, grad in zip(self.velocities,grads)]
		updates = self.velocities
		if nesterov:
			updates = [self.momentum * velocity - self.learning_rate * grad for velocity, grad in zip(self.velocities,grads)]

		return updates


class AdamOptimizer(BaseOptimizer):

	def __init__(self, params, learning_rate_init=0.1, beta1=0.9, beta2=0.99, epsilon=10e-8):
		super(AdamOptimizer, self).__init__(params, learning_rate_init)

		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.velocities = [np.zeros_like(param) for param in params]
		self.caches = [np.ones_like(param) for param in params]
		self.t = 0


	def _get_updates(self, grads):

		self.t += 1 
		self.velocities = [(beta1 * velocity + (1 - beta1) * grad) for velocity, grad in zip(self.velocities,grads)]
		self.caches = [(beta2 * cache + (1 - beta2) * grad ** 2) for cache, grad in zip(caches,grads)]
		self.learning_rate = self.learning_rate_init * (np.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t))
		updates = [-self.learning_rate * velocity / (np.sqrt(cache) + self.epsilon) for velocity, cache in zip(self.velocities,self.caches)]

		return updates

