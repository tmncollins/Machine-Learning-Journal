import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LogisticRegressor:

    def __init__(self, theta=[]):
        self.theta = theta

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_gradient(self, X, y):
        m = y.size
        return (X.T @ (self.sigmoid(X @ self.theta) - y) / m)

    # gradient descent
    def train(self, X, y, L=0.1, iterations=1000, tol=1e-7):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X_b.shape[1])
        print(self.theta, X)

        for i in range(iterations):
            gradient = self.sigmoid_gradient(X_b, y)
            self.theta -= L * gradient

            if np.linalg.norm(gradient) < tol:
                break

    def predict_probability(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_probability(X) >= threshold).astype(int)

data = pd.read_csv('Fish.csv')

bream = data.query('Species == "Bream"')
perch = data.query('Species == "Pike"')
smelt = data.query('Species == "Smelt"')
fish = data.query('Species == "Pike" | Species == "Bream"')
X = fish.drop("Species", axis=1)
for DROP in ['Length2', 'Length3', 'Height', 'Width']:
    X.drop(DROP, axis=1, inplace=True)
y = fish["Species"].map({"Pike": 1, "Bream": 0})


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegressor()
clf.train(X_train_scaled, y_train)
print(clf.theta)
y_predict_train = clf.predict(X_train_scaled)
y_predict_test = clf.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

train_acc = accuracy_score(y_train, y_predict_train)
test_acc = accuracy_score(y_test, y_predict_test)

print(train_acc)
print(test_acc)

m = -clf.theta[2] / float(clf.theta[1])
b = -clf.theta[0] / float(clf.theta[1])
fish_scaled = scaler.transform(X)
fish_y = fish_scaled[:, 0]
fish_x = fish_scaled[:, 1]
MIN=  min(fish_x)
MAX = max(fish_x)
plt.plot([x for x in [MIN,MAX+1]], [m*x+b for x in [MIN, MAX+1]])
plt.scatter(fish_x, fish_y, c=y)
#plt.legend(loc="best")
plt.title("Logistic Regression: Bream vs Smelt")
plt.xlabel("Length")
plt.ylabel("Weight")
plt.xticks([], [])
plt.yticks([], [])
print(m ,b)
plt.show()

