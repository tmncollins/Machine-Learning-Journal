import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import numpy as np

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
SYNTHETIC_SCALER = 3

def bias_dataset(X_t, y_t, max_occur):
    counter = [0 for i in range(10)]
    X_train = []
    y_train = []
    for i in range(len(y_t)):
        d = y_t[i]
        if counter[d] < max_occur[d]:
            counter[d] += 1
            X_train.append(X_t[i])
            y_train.append(d)
    return X_train, y_train

def gaussian_noise(image, mean, std):
    noise = np.random.normal(mean, std, (image.shape[0], image.shape[1]))
    return image + noise

def normalise(image):
    return image / image.max()

X_train = X_t
y_train = y_t

if not FULL_DATASET:
    X_train, y_train = bias_dataset(X_t, y_t, [10 for i in range(10)])

X_synth = []
y_synth = []

for img, label in zip(X_train, y_train):
    for i in range(SYNTHETIC_SCALER):
        y_synth.append(label)
        synth_img = normalise(gaussian_noise(img.reshape(8,8), 0.8, 0.2).flatten())
        X_synth.append(synth_img)


X_train = X_train + X_synth
y_train = y_train + y_synth
# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(clf.score(X_test, y_test))

_, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
axes = list(np.array(axes.ravel()))
axes = [(axes[0], axes[1], axes[2]), (axes[3], axes[4], axes[5]), (axes[6], axes[7], axes[8])]
for ax, image, prediction, answer in zip(axes, X_test, predicted, y_test):
    print(ax)
    ax1 = ax[0]
    ax1.set_axis_off()
    image = image.reshape(8, 8)
    ax1.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax1.set_title(f"Original")
    ax2 = ax[1]
    ax2.set_axis_off()
    image = image.reshape(8, 8)
    image = normalise(gaussian_noise(image.reshape(8,8), 0.2, 0.1))
    ax2.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax2.set_title(f"Low Noise")
    ax3 = ax[2]
    ax3.set_axis_off()
    image = image.reshape(8, 8)
    image = normalise(gaussian_noise(image.reshape(8,8), 0.8, 0.2))
    ax3.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax3.set_title(f"High Noise")

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