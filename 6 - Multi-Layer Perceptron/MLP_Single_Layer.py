import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

class MLP_Classifier:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_layer_size)
        self.weights_hidden_output = np.random.randn(hidden_layer_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_layer_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def softmax(self, X):
        exp_x = np.exp(X - np.max(X))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward_prop(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def backward_prop(self, X, y, L=0.01):
        output = self.forward_prop(X)
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)

        self.weights_hidden_output -= L * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= L * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= L * np.dot(X.T, hidden_error)
        self.bias_hidden -= L * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000, L=0.1):
        for epoch in range(epochs):
            self.backward_prop(X, y, L)
            if (epoch+1) % 100 == 0:
                output = self.forward_prop(X)
                loss = -np.sum(y * np.log(output)) / X.shape[0]
                print(epoch+1, loss)

    def predict(self, X):
        output = self.forward_prop(X)
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
clf = MLP_Classifier(len(X[0]), 100, len(SPECIES))
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

plt.show()
