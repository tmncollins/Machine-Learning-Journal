# IMPORT PACKAGES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# IMPORT DATASET
df1 = pd.read_csv('Iris.csv')
df1.head()

# PROCESS DATASET
df1['Species'] = df1['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
df1.drop(['Id'],axis=1,inplace=True)

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn import svm


rdm = 100
svm_linear = SVC(kernel="linear", gamma=2, C=1, random_state=rdm)
svm_rbf = SVC(gamma=2, C=1, random_state=rdm)
svm_polynomial = SVC(kernel="poly", gamma=2, C=1, random_state=rdm)
svm_sigmoid = SVC(kernel="sigmoid", gamma=2, C=1, random_state=rdm)

X = df1.drop(["Species"],axis=1).values
y = df1["Species"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

svm_linear.fit(X_train, y_train)
print("SVM Linear Score:  ", svm_linear.score(X_test, y_test))
svm_rbf.fit(X_train, y_train)
print("SVM RBF Score:  ", svm_rbf.score(X_test, y_test))
svm_sigmoid.fit(X_train, y_train)
print("SVM Sigmoid Score:  ", svm_sigmoid.score(X_test, y_test))
svm_polynomial.fit(X_train, y_train)
print("SVM Polynomial Score:  ", svm_polynomial.score(X_test, y_test))

def plot_training_data_with_decision_boundary(
    kernel, ax=None, long_title=True, support_vectors=True
):
    # Train the SVC
    clf = SVC(kernel=kernel, gamma=2).fit(X_train[:,2:4], y_train)

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X_train[:, 2:4], "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    """    
    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )
    """

    # Plot samples by color and add legend
    ax.scatter(X_train[:, 2], X_train[:, 3], c=y_train, s=30, edgecolors="k")
#    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)

    ax.set_xlabel("Petal Length / cm")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("Petal Width / cm")

    plt.show()

plot_training_data_with_decision_boundary(kernel="linear")
plot_training_data_with_decision_boundary(kernel="rbf")
plot_training_data_with_decision_boundary(kernel="sigmoid")
plot_training_data_with_decision_boundary(kernel="poly")