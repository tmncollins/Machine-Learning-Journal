from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import rdkit

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from mordred import Calculator, descriptors

import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem
import pickle

import warnings
warnings.filterwarnings("ignore")

# define function that transforms SMILES strings into ECFPs
def ECFP_from_smiles(smiles, R=2, L=2 ** 10, use_features=False, use_chirality=False):
    """
    Inputs:

    - smiles ... SMILES string of input compound
    - R ... maximum radius of circular substructures
    - L ... fingerprint-length
    - use_features ... if false then use standard DAYLIGHT atom features, if true then use pharmacophoric atom features
    - use_chirality ... if true then append tetrahedral chirality flags to atom features

    Outputs:
    - np.array(feature_list) ... ECFP with length L and maximum radius R
    """

    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=R,
                            nBits=L, useFeatures=use_features, useChirality=use_chirality)
    return np.array(feature_list)

def getMolDescriptors(mol, missingVal=0):
    res = {}
    mol = Chem.MolFromSmiles(mol)
    for nm, fn in Descriptors._descList:
        # some of the descriptor functions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
#            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


def getMordredDescriptors(smiles):
    calc = Calculator(descriptors, ignore_3D=False)
    mol = Chem.MolFromSmiles(smiles)

    # pandas df
    res = dict(calc(mol))
    return res


# IMPORT DATASET - FINGERPRINT
df_train = (pd.read_csv('/kaggle/input/smiles-boiling-point/bp_data.csv'))
print(len(df_train))

X = df_train.drop('bp_K', axis=1)
fingerprint_length = 2**11
fingerprint_cols = [[] for _ in range(fingerprint_length)]
for f in df_train['fingerprint']:
    f = str(f)
    zeros = fingerprint_length - len(f)
    f = ("0" * zeros) + f
    for i in range(fingerprint_length):
        if f[i] == "1": fingerprint_cols[i].append(1)
        else: fingerprint_cols[i].append(0)

print(len(X.columns))

# CUT OFF COLUMNS WITH LOW VARIATION
min_spread = 0.001
for i in range(fingerprint_length):
    if min_spread <= (fingerprint_cols[i].count(1) / len(fingerprint_cols[i])) <= 1 - min_spread:
        X['fp_' + str(i)] = fingerprint_cols[i]

print(len(X.columns))

SCALER = 2000
X = X.copy()
X = X.drop('smiles', axis=1).drop('fingerprint', axis=1)
X['mw'] = np.log(X['mw'])
X['valence e'] = np.log(X['valence e'])
y = np.log(np.array(df_train['bp_K'].values / SCALER, dtype=float))
input_size = len(X.columns)
_X = X.copy()
_X['bp_K'] = np.log(df_train['bp_K'].values)
print(X.head())

X_FINGERPRINT = X

# IMPORT DATASET - DESCRIPTORS
df_train = (pd.read_csv('/kaggle/input/smiles-boiling-point/bp_descriptors.csv'))
print(len(df_train))

SMILES = df_train["smiles"]
X = df_train.drop('bp_K', axis=1).drop('smiles', axis=1)
print(len(X.columns))

# CUT OFF COLUMNS WITH LOW VARIATION
min_spread = 0.01
TO_DROP = []
for col in X.columns:
    if max(X[col].value_counts()) / len(X[col]) >= 1 - min_spread:
        TO_DROP.append(col)

for col in TO_DROP:
    X = X.drop(col, axis=1)

print(len(X.columns))

scaler = StandardScaler()

COLUMNS = X.columns
SCALER = 2000
X = X.copy().select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
y = df_train['bp_K'].values / SCALER
#y = np.log(np.array(df_train['bp_K'].values / SCALER, dtype=float))
input_size = len(X.columns)
_X = X.copy()
_X['bp_K'] = df_train['bp_K'].values
print(X.head())

for c in X_FINGERPRINT.columns.values:
    if c not in _X.columns.values:
        _X[c] = X_FINGERPRINT[c]

X = np.array(X.values, dtype=float)

numeric_cols = _X.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns
numeric_cols = numeric_cols.drop('bp_K')
corr = _X[numeric_cols].corrwith(_X['bp_K']).sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=corr.index, y=corr.values)
plt.xticks(rotation=90)
plt.title("Correlation of numeric features with boiling point")
plt.show()

SMILES = df_train["smiles"]

y = np.log(df_train['bp_K'].values)
#y = df_train['bp_K'].values / SCALER
print(X.shape)
print(y.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X)
SMILES_TO_DESCRIPTORS = dict()
for row, s in zip(X, SMILES):
    SMILES_TO_DESCRIPTORS[s] = row
train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1)
print(train_x.shape)

train_dataset = []
for a, b in zip(train_x, train_y):
    train_dataset.append((a,b))

val_dataset = []
for a, b in zip(val_x, val_y):
    val_dataset.append((a,b))

SMILES_TO_BP = dict()
for bp, s in zip(y, SMILES):
    SMILES_TO_BP[s] = bp

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=-1
)

param_dist = {
    'n_estimators':[200,300,400,500],
    'max_depth': [3,4,5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}


#model = XGBRegressor()
#xgb = RandomizedSearchCV(model, param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=10, verbose=True)
xgb_model = xgb.fit(train_x, train_y)

print(xgb_model.score(train_x, train_y))
print(xgb_model.score(val_x, val_y))
# Parameter which gives the best results
#print(f"Best Hyperparameters: {xgb.best_params_}")
# Accuracy of the model after using best parameters
#print(f"Best Score: {xgb.best_score_}")

from sklearn.ensemble import RandomForestRegressor

def train_rf(n_estimators, train_x, train_y, random_state=0):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(train_x, train_y)

    return rf_model

train_score = []
val_score = []
forest_size = [10, 20, 30, 40, 50]
for n in forest_size:
    print("=== n_estimators: " + str(n) + " ===")
    rf = train_rf(n, train_x, train_y)
    train_s = rf.score(train_x, train_y)
    val_s = rf.score(val_x, val_y)
    print(train_s, val_s)
    train_score.append(train_s)
    val_score.append(val_s)

plt.plot(forest_size, train_score, label="train")
plt.plot(forest_size, val_score, label="val")
plt.legend("loc=best")
plt.show()

plt.plot(forest_size, train_score, label="training")
plt.plot(forest_size, val_score, label="validation")
plt.ylabel('Loss')
plt.xlabel('Forest Size')
plt.legend(loc='best')
plt.show()

rf_model = train_rf(10, train_x, train_y)
print(rf_model.score(train_x, train_y))
print(rf_model.score(val_x, val_y))

chosen_model = rf_model

scatter_x = []
scatter_y = []
scatter_val_x = []
scatter_val_y = []
for molecule, bp in tqdm(zip(train_x, train_y), total=min(len(train_x), len(train_y))):
    molecule = molecule.reshape(1,-1)
    pred_bp = chosen_model.predict(molecule)
    scatter_x.append(pred_bp)
    scatter_y.append(bp)
for molecule, bp in tqdm(zip(val_x, val_y), total=min(len(val_x), len(val_y))):
    molecule = molecule.reshape(1,-1)
    pred_bp = chosen_model.predict(molecule)
    scatter_val_x.append(pred_bp)
    scatter_val_y.append(bp)

scatter_x = np.exp(scatter_x)
scatter_y = np.exp(scatter_y)
scatter_val_x = np.exp(scatter_val_x)
scatter_val_y = np.exp(scatter_val_y)

#scatter_x = SCALER * np.array(scatter_x)
#scatter_y = SCALER * np.array(scatter_y)
#scatter_val_x = SCALER * np.array(scatter_val_x)
#scatter_val_y = SCALER * np.array(scatter_val_y)


plt.scatter(scatter_x, scatter_y, label='training', s=5, alpha=0.5)
plt.scatter(scatter_val_x, scatter_val_y, label='validation', s=5, alpha=0.5)
plt.plot(scatter_y, scatter_y, c="black", linewidth=0.5)
plt.plot(scatter_y, 1.1* np.array(scatter_y), c="black", linestyle="dashed", linewidth=0.5)
plt.plot(scatter_y, 0.9* np.array(scatter_y), c="black", linestyle="dashed", linewidth=0.5)
plt.xlabel("Predicted BP / K")
plt.ylabel("Actual BP / K")
plt.legend(loc="best")
plt.show()

molecules = []
SMILES_SET = set(SMILES)
below_10 = 0
for mol in tqdm(SMILES_SET):
    x = SMILES_TO_DESCRIPTORS[mol]
#    x = train_x[mol]
    pred = np.exp(rf_model.predict([x])[0])
    bp = np.exp(SMILES_TO_BP[mol])
    diff = 100 * abs(bp - pred) / bp
    if diff < 1: below_10 += 1
    molecules.append((diff, pred, bp, mol))
molecules = sorted(molecules)[::-1]
for i in range(10):
    print(molecules[i])
print(below_10, below_10 / len(molecules))
