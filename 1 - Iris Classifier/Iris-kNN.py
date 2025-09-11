# IMPORT PACKAGES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# IMPORT DATASET
df1 = pd.read_csv('Iris.csv')
df1.head()

def get_info_dataframe(dataframe):
    print(f"DATAFRAME GENERAL INFO - \n")
    print(dataframe.info(),"\n")
    print(f"DATAFRAME MISSING INFO - \n")
    print(dataframe.isnull().sum(),"\n")
    print(f"DATAFRAME SHAPE INFO - \n")
    print(dataframe.shape)

# EVALUATE DATASET
get_info_dataframe(df1)
df1['Species'].unique()

# PROCESS DATASET
df1['Species'] = df1['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
df1.drop(['Id'],axis=1,inplace=True)
df1.head()

from sklearn.neighbors import KNeighborsClassifier

X = df1.drop(["Species"],axis=1).values
y = df1["Species"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

print("k   |  Score")
print("----+--------")
y_values = []
for k in range(1, 21):
    k_neigh = KNeighborsClassifier(n_neighbors=k)
    k_neigh.fit(X_train, y_train)
    s = k_neigh.score(X_test, y_test)
    print(k, "  | ", "{:.5f}".format(s))
    y_values.append(s)

import matplotlib.pyplot as plt
import numpy as np


plt.plot([i for i in range(1, 21)], y_values, color = 'r')
plt.xlabel("k")
plt.ylabel("Score")
plt.title("k-NN Classifiers")
plt.show()


import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

ax = plt.subplot()
k_neigh = KNeighborsClassifier(n_neighbors=10)

k_neigh.fit(X_train[:, 2:4], y_train)

disp = DecisionBoundaryDisplay.from_estimator(
    k_neigh,
    X_test[:, 2:4],
    response_method="predict",
    plot_method="pcolormesh",
    xlabel="Petal Length / cm",
    ylabel="Petal Width / cm",
    shading="auto",
    alpha=0.5,
    ax=ax,
)
scatter = disp.ax_.scatter(X[:, 2], X[:, 3], c=y, edgecolors="k")
disp.ax_.legend(
    scatter.legend_elements()[0],
    ["Setosa", "Versicolor", "Virginica"],
    loc="lower left",
    title="Classes",
)
_ = disp.ax_.set_title(
    f"k-NN Classification\n(k = 10)"
)

plt.show()