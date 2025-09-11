import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import *
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Fish.csv')

bream = data.query('Species == "Bream"')
pike = data.query('Species == "Pike"')
smelt = data.query('Species == "Smelt"')
plt.scatter(bream['Length1'], bream['Weight'], label='Bream')
plt.scatter(pike['Length1'], pike['Weight'], label='Pike')
plt.scatter(smelt['Length1'], smelt['Weight'], label='Smelt')
plt.xlabel('Length / cm')
plt.ylabel('Weight / g')
plt.legend(loc='best')
plt.show()

class LinearRegressor:
    def __init__(self, theta=[]):
        self.theta = theta

    def loss_function_squared(self, X, y):
        total_error = 0
        for i in range(len(X)):
            _x = X[i]
            _y = y[i]
            total_error += (_y - (X @ self.theta + self.const))**2
        return total_error / len(X)

    def loss_function_logcosh(self, m, b, x, y):
        total_error = 0
        for i in range(len(x)):
            _x = x[i]
            _y = y[i]
            total_error += log(cosh(m*_x+b-_y))
        return total_error / len(x)

    def loss_function_absolute(self, m, b, x, y):
        total_error = 0
        for i in range(len(x)):
            _x = x[i]
            _y = y[i]
            total_error += abs(_y - (m*_x+b))
        return total_error / len(x)

    def MSE_gradient(self, X, y):
        n = len(X)
#        print(self.theta, X)
        return -2 * (y - (X @ self.theta + self.const))

    # Gradient Descent for MSE
    def train(self, X, y, L=0.001, iterations=100000, tol=1e-7):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.ones(X_b.shape[1])

        for i in range(iterations):
            gradient = X_b.T @ (y - (X_b @ self.theta))
#            print(self.theta, gradient)
            self.theta += L * gradient

            print(np.linalg.norm(gradient))
            if np.linalg.norm(gradient) < tol:
                print("tolerance")
                break


    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.T @ self.theta

epochs = 100000
L = 0.1
m = 0
b = 0
X = bream[['Length1', 'Length2', 'Length3']]
#X = bream['Length1'].values.reshape(-1,1)
Y = bream['Weight'].values

#print(X, Y)

loss_list = []

scaler = StandardScaler()
#X = np.log(X)
#Y = np.log(Y)
X = scaler.fit_transform(X)

reg = LinearRegressor()
reg.train(X, Y)

# Bream Epochs: 108000
# Pike Epochs:  338000
# Smelt Epoch:  140000

"""
plt.plot(loss_list)
plt.xlabel("epochs")
plt.title("Smelt Linear Regression Error")
plt.ylabel("Mean Squared Error")
plt.show()
"""

plt.scatter(X[:, 0], Y)
b = reg.theta[0]
m = reg.theta[-1]
MIN = min(X[:, 0])
MAX = max(X[:, 0])
print(m, b)
plt.plot([MIN, MAX], [m * i + b for i in [MIN, MAX]])
plt.xlabel("Length / cm")
plt.ylabel("Weight / g")
plt.title("Bream Linear Regression")
plt.xticks([], [])
plt.show()

#
# MEAN ABSOLUTE
# 52.26592853161895 -965.1832090164291 5250.693557266451
# MEAN SQUARED
# 52.26592853161895 -965.1832090164291 5250.693557266451
# ROOT-MEAN SQUARED
# 52.26592853161895 -965.1832090164291 5250.693557266451
# LOG-COSH
# 52.26592853161895 -965.1832090164291 5250.693557266451