# IMPORT PACKAGES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from sklearn import svm

# IMPORT DATASET
df_train = pd.read_csv('Titanic-Dataset.csv')

MALE = 0
FEMALE = 1
CHILD = 0
ADULT = 1

# PROCESS DATASET
# ENCODE CATEGORICALS
df_train['Sex'] = df_train['Sex'].map({'male':MALE,'female':FEMALE})
df_train['Embarked'] = df_train['Embarked'].map({'S':0,'C':1,'Q':2})
ages = []
for i, row in df_train.iterrows():
    age = row['Age']
    if age:
        if age < 16: ages.append(CHILD)
        else: ages.append(ADULT)
    else:
        if "Master" or "Miss" in row['Name']: ages.append(CHILD)
        else: ages.append(ADULT)
df_train['Age'] = ages
df_train['Surname'] = [i.split(",")[0] for i in df_train['Name']]
NAMES = list(set(df_train['Name']))
# DROP UNNECESSARY IDENTIFIERS
TO_DROP = ['PassengerId', 'Cabin', 'Embarked', 'Ticket']
for DROP in TO_DROP:
    df_train.drop([DROP],axis=1,inplace=True)
df_train.dropna(inplace=True)

from collections import *
family_survive = defaultdict(int)
family_die = defaultdict(int)

for i, row in df_train.iterrows():
    if row['Sex'] == FEMALE or row['Age'] == CHILD:
        if row['Survived']: family_survive[row['Surname']] += 1
        else: family_die[row['Surname']] += 1
    else:
        if not row['Survived']: family_die[row['Surname']] += 1

TO_DROP = ['Name']
for DROP in TO_DROP:
    df_train.drop([DROP],axis=1,inplace=True)
df_train.dropna(inplace=True)

def predict(passenger):
    if passenger['Sex'] == MALE:
        if passenger['Age'] == CHILD:
            if family_survive[passenger['Surname']] + family_die[passenger['Surname']] == 0:
                print(passenger['Surname'])
                return False
            if family_die[passenger['Surname']] / (family_survive[passenger['Surname']] + family_die[passenger['Surname']]) < 0.2: return True
        else:
            return False
            return svm_linear.predict(passenger.drop('Surname').values.reshape(1,-1))[0]
    else:
        if passenger['Pclass'] in [1,2]: return True
        if family_survive[passenger['Surname']] + family_die[passenger['Surname']] == 0:
            return True
        if family_survive[passenger['Surname']] / (family_survive[passenger['Surname']] + family_die[passenger['Surname']]) > 0.2: return True
    return False

score = 0
tot = 0
survived = 0
for i, row in df_train.iterrows():
    print(i, row.values)
    s = predict(row.drop('Survived'))
    if s == row['Survived']: score += 1
    tot += 1
    survived += row['Survived']
print(score / tot)
print(survived / tot)

print("TESTING")
# IMPORT DATASET
df_test = pd.read_csv('Titanic-Dataset-Test.csv')
#df_ans = pd.read_csv('correct_ans.csv')
#ANSWER = dict()
#for i, row in df_ans.iterrows():
#    ANSWER[row['PassengerId']] = row['Survived']
# PROCESS DATASET
# ENCODE CATEGORICALS
df_test['Sex'] = df_test['Sex'].map({'male':0,'female':1})
df_test['Embarked'] = df_test['Embarked'].map({'S':0,'C':1,'Q':2})
ages = []
for i, row in df_test.iterrows():
    age = row['Age']
    if age:
        if age < 16: ages.append(CHILD)
        else: ages.append(ADULT)
    else:
        if "Master" or "Miss" in row['Name']: ages.append(CHILD)
        else: ages.append(ADULT)
df_test['Age'] = ages
df_test['Surname'] = [i.split(",")[0] for i in df_test['Name']]
# DROP UNNECESSARY IDENTIFIERS
TO_DROP = ['Cabin', 'Embarked', 'Ticket', 'Parch', 'Name']
for DROP in TO_DROP:
    df_test.drop([DROP],axis=1,inplace=True)

df_test.fillna(0, inplace=True)

survived = 0
surv = 0
total = 0
CORRECT = 0
df_out = pd.DataFrame(columns=['PassengerId','Survived'])
for i, row in df_test.iterrows():
    r = row.drop('PassengerId')
    s = int(predict(row))
    survived += s
#    surv += ANSWER[row['PassengerId']]
    total += 1
#    if s == ANSWER[row['PassengerId']]: CORRECT += 1
#    print(row["PassengerId"], s)
    df_out = df_out._append({'PassengerId': int(row["PassengerId"]), 'Survived':s}, ignore_index=True)
print(df_out)
df_out.to_csv('ans.csv', index=False)
print(survived / total)
print(surv / total)
print(CORRECT / total)
