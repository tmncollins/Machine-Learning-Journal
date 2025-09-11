import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_t, X_test, y_t, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True
)
FULL_DATASET = False

X_train = []
y_train = []
if FULL_DATASET:
    X_train = X_t
    y_train = y_t

else:
    size = [50, 50, 50, 140, 50, 50, 50, 50, 50, 50]
    counter = [0 for i in range(10)]

    for i in range(len(y_t)):
        d = y_t[i]
        if counter[d] < size[d] // 2:
            counter[d] += 1
            X_train.append(X_t[i])
            y_train.append(d)


# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 9))
axes = axes.ravel()
print(axes)
for ax, image, prediction, answer in zip(axes, X_test, predicted, y_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

plt.hist(y_train, bins=[0,1,2,3,4,5,6,7,8,9,10], density=False, rwidth=0.75)
plt.xlabel("Digit")
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5], [0,1,2,3,4,5,6,7,8,9])
plt.ylabel("Frequency")
plt.show()