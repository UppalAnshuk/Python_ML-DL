# m,X_b,y need to be initialised to run this routine

import numpy as np

n_epochs = 50

t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t): ##responsible for determining the learning parameter
	return t0 / (t + t1)
theta = np.random.randn(2,1) # random initialization of parameters
for epoch in range(n_epochs):
	for i in range(m):
		random_index = np.random.randint(m) #picking any random training example
		xi = X_b[random_index:random_index+1] #X_b is the training dataset
		yi = y[random_index:random_index+1]
		gradients = 2 * xi.T.dot(xi.dot(theta) - yi) #gradients for the descent
		eta = learning_schedule(epoch * m + i) #learning rate decreases as epoch and i increase
		theta = theta - eta * gradients
		
		##The learning schedule will keep decreasing the learning schedule so that the learning rate intially is large and we jump around till we hopefully reach the global minima, then the learning rate decreases and aviods sub-optimal convergence
