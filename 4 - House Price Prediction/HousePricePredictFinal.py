import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

train_df = pd.read_csv("HousePrice-Train.csv")
test_df = pd.read_csv("HousePrice-Test.csv")

def preprocess(df):
    # LotFrontage – median
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Numeric - NA means no feature → 0
    num_zero = ["MasVnrArea", "GarageYrBlt", "BsmtFullBath", "BsmtHalfBath",
                "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "GarageArea"]
    for col in num_zero: df[col] = df[col].fillna(0)

    # Categorical - NA means "None"
    cat_none = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual", "Electrical",
        "MSZoning", "Utilities", "Functional", "Exterior1st", "Exterior2nd", "KitchenQual", "SaleType"
    ]
    for col in cat_none: df[col] = df[col].fillna("None")

    # Feature Engineering
    df['IsNew'] = [1 if i == 'New' else 0 for i in df['SaleType']]
    df['Is2Stories'] = [1 if i == '2Story' else 0 for i in df['HouseStyle']]
    df['Is2Fam'] = [1 if i == '2fmCon' else 0 for i in df['BldgType']]
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = (df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    # House age
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['HouseAge'] = [max(0, i) for i in df['HouseAge']]
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['RemodAge'] = [max(0, i) for i in df['RemodAge']]
    df['QualityArea'] = df['OverallQual'] * df['GrLivArea']

    # Categorical - One Hot Encoding
    to_encode = ['MSSubClass', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
                 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'MSZoning']
    for ohe in to_encode:
        df = df.join(pd.get_dummies(df[ohe], prefix=ohe, dtype=int)).drop(ohe, axis=1)

    return df

y_train_raw = train_df['SalePrice']
x_train_raw = train_df.drop("Id", axis=1)
x_test_raw = test_df.copy()

# Preprocess
x_train = preprocess(x_train_raw)
x_test = preprocess(x_test_raw)

# Remove unused features
x_train = x_train.drop(['SalePrice'], axis=1)
x_test_to_remove = []
for i in x_test.columns:
    if i not in x_train.columns: x_test_to_remove.append(i)
for i in x_test_to_remove:
    x_test = x_test.drop(i, axis=1)

x_train_to_remove = []
for i in x_train.columns:
    if i not in x_test.columns: x_train_to_remove.append(i)
for i in x_train_to_remove:
    x_train = x_train.drop(i, axis=1)

# Hyperparameter tuning - CV Search
param_dist = {'n_estimators':[200,300,400,600,800,1000],
              'max_depth':[2,3,4,5],
              'subsample':[0.4,0.6,0.8,0.9,1.0],
              'learning_rate':[0.05,0.01,0.1,0.2],
              'colsample_bytree':[0.4,0.6,0.8,0.9,1.0]}

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

y_train = y_train_raw
#model = XGBRegressor()
#xgb = RandomizedSearchCV(model, param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=100, verbose=True)
xgb_model = xgb.fit(x_train.values, y_train.values)

#print(f"Best Hyperparameters: {xgb.best_params_}")
#print(f"Best Score: {xgb.best_score_}")

# Test: 80 columns
y_test = xgb_model.predict(x_test)

# Add predicts to dataset of test
test_out_df = test_df.copy()
test_out_df["SalePrice"] = y_test

test_out_df=test_out_df[["Id","SalePrice"]]

#test1_df.to_csv("results_with_rf.csv", index=False)
test_out_df.to_csv("ans_xgb.csv", index=False)