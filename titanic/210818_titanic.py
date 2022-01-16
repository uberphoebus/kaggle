
# --------------------------------------------------
# import
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore')

def nan_check():
    """missing val check frame"""
    nan_dict = {'CNT':train_test.isna().sum(),
                'RATE':train_test.isna().sum()
                       / train_test.shape[0] * 100}
    nan_df = pd.DataFrame(nan_dict)
    return nan_df.head().T

# data read
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
submission = pd.read_csv('./input/gender_submission.csv')

# train_test append
train_test = train.append(test)

# --------------------------------------------------
# nan value
# --------------------------------------------------

# Sex col
train_test['Sex'] = train_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Embarked
train_test['Embarked'] = train_test['Embarked'].fillna('S')
train_test['Embarked'] = train_test['Embarked'].apply(lambda x: 0 if x == 'S'else (1 if x == 'C' else 2))

# Name_title
train_test['Name_title'] = train_test['Name'].str.extract('([A-Za-z]+)\.')
title_map = {'Mr':0,
             'Miss':1, 'Mlle':1, 'Ms':1, 'Lady':1,
             'Mrs':2, 'Mme':2,
             'Master':3, 'Dr':3, 'Sir':3,
             'Rev':4, 'Col':4, 'Major':4, 'Don':4,
             'Capt':4, 'Countess':4, 'Jonkheer':4, 'Dona':4}
train_test['Name_title'] = train_test['Name_title'].map(title_map)
age_gr = train_test.groupby('Name_title')['Age']

# Age
train_test['Age'] = train_test['Age'].fillna(age_gr.transform('mean'))
train_test['Age'] = np.round(train_test['Age'], 0).astype('int64')

# family = SibSp + Parch
train_test['family'] = train_test['SibSp'] + train_test['Parch'] + 1
fam_pc_cond = train_test[(train_test['family'] == 1) & (train_test['Pclass'] == 3)]

# Fare
mean_fare = fam_pc_cond['Fare'].mean()
train_test['Fare'] = train_test['Fare'].fillna(mean_fare)
mean_fare = round(mean_fare, 4)

# binding Age
train_test.loc[train_test['Age'] <= 16 , 'Age_band1'] = 1
train_test.loc[(train_test['Age'] > 16) & (train_test['Age'] <= 34), 'Age_band1'] = 2
train_test.loc[(train_test['Age'] > 34) & (train_test['Age'] <= 48), 'Age_band1'] = 3
train_test.loc[(train_test['Age'] > 48) & (train_test['Age'] <= 67), 'Age_band1'] = 4
train_test.loc[train_test['Age'] > 67 , 'Age_band1'] = 5
train_test['Age_band1'] = train_test['Age_band1'].astype('int64')
train_test['Age_band2'] = pd.cut(train_test['Age'],bins=[0, 16, 34, 48, 67, 90],labels=[1, 2, 3, 4, 5])
train_test['Age_band3'] = train_test['Age'] // 10

# --------------------------------------------------
# drop cols
# --------------------------------------------------
drop_cols = ['Name', 'Age', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age_band2', 'Age_band3']
train_test.drop(drop_cols, axis=1, inplace=True)

# --------------------------------------------------
# train_test_split
# --------------------------------------------------
train = train_test[~train_test['Survived'].isna()]
test  = train_test[train_test['Survived'].isna()]
train['Survived'] = train['Survived'].astype('int64')
test.drop('Survived', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# --------------------------------------------------
# DecisionTreeClassifier
# --------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_val)

from sklearn.metrics import accuracy_score
dt_acc_score = accuracy_score(y_val, dt_pred)

# --------------------------------------------------
# RandomForestClassifier
# --------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
rf_acc_score = accuracy_score(y_val, rf_pred)

# --------------------------------------------------
# submission
# --------------------------------------------------
rf_pred = rf_model.predict(test)
submission['Survived'] = rf_pred.reshape(-1, 1)
submission.to_csv('./titanic_sub_v01.csv', index=False)
df = pd.read_csv('./titanic_sub_v01.csv')

# --------------------------------------------------
# feature importance check
# --------------------------------------------------
rf_fi = rf_model.feature_importances_ * 100
rf_cols = test.columns.values
df = pd.DataFrame(rf_fi, index=rf_cols)
df = df.sort_values(by=0, ascending=False)

dict = {'feature':rf_cols, 'imp':rf_fi}
impdf = pd.DataFrame(dict).sort_values(by='imp', ascending=False).T
print(impdf)
