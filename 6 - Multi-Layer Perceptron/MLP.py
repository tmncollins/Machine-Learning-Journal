import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

class MLP_Classifier:
    def __init__(self, input_size, output_size, hidden_layer_sizes=[100]):
        layer_size = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.nlayers = len(layer_size)
        self.weights = [np.random.randn(layer_size[i], layer_size[i+1]) for i in range(self.nlayers - 1)]
        self.bias = [np.zeros((1, layer_size[i])) for i in range(1, self.nlayers)]

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def softmax(self, X):
        exp_x = np.exp(X - np.max(X))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward_prop(self, X):
        self.vectors = [X]
        for i in range(self.nlayers - 1):
            v_in = np.dot(self.vectors[-1], self.weights[i]) + self.bias[i]
            if i == self.nlayers - 2:
                v_out = self.softmax(v_in)
            else:
                v_out = self.sigmoid(v_in)
            self.vectors.append(v_out)
        return self.vectors[-1]

    def backward_prop(self, X, y, L=0.01):
        output = self.forward_prop(X)
        errors = [output - y]
        for i in range(self.nlayers - 1):
            idx = -(i+1)
            next_error = np.dot(errors[-1], self.weights[idx].T) * self.vectors[idx-1] * (1 - self.vectors[idx-1])
            self.weights[idx] -= L * np.dot(self.vectors[idx-1].T, errors[-1])
            self.bias[idx] -= L * np.sum(errors[-1], axis=0, keepdims=True)
            errors.append(next_error)

    def train(self, X, y, epochs=1000, L=0.01):
        for epoch in range(epochs):
            self.backward_prop(X, y, L)
            if (epoch+1) % 1000 == 0:
                output = self.forward_prop(X)
                loss = -np.sum(y * np.log(output)) / X.shape[0]
                print(epoch+1, loss)

    def predict(self, X):
        output = self.forward_prop(X)
        print(output)
        return np.argmax(output, axis=1)

df = pd.read_csv('Fish.csv')
SPECIES = df['Species'].unique()
df['Species'] = df['Species'].map({SPECIES[i]: i for i in range(len(SPECIES))})
X = df.drop('Species', axis=1 ).values
y = []
for s in df['Species']:
    line = [0 for _ in range(len(SPECIES))]
    line[s] = 1
    y.append(line)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("training")
clf = MLP_Classifier(len(X[0]), len(SPECIES), [10, 10, 10])
print(y_train)
clf.train(X_train, y_train)
print("trained")

ans = 0
predicted = []
for x, y in zip(X_test, y_test):
    pred_y = clf.predict(x)[0]
    print(pred_y, y)
    if y[pred_y] == 1: ans += 1
    predicted.append(SPECIES[pred_y])

print(ans / len(y_test))

Y = []
for line in y_test: Y.append(SPECIES[line.index(1)])
print(Y, predicted)

disp = metrics.ConfusionMatrixDisplay.from_predictions(Y, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

#print(clf.vectors, clf.weights, clf.bias)

plt.show()
