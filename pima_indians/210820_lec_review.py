
# ------------------------------------------------
# import
# ------------------------------------------------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/02_pima/input/diabetes.csv')


# ------------------------------------------------
# data check
# ------------------------------------------------
print(df.head())
print(df.describe())
print(df.shape)
print(df.columns)
print(df.info())
sns.set()
df.hist(figsize=(12, 8))
# plt.show()


# ------------------------------------------------
# def split_fit_score
# ------------------------------------------------
def split_fit_score(X, y, model=None, test_size=0.2, random_state=1414, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=shuffle)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('accuracy_score = ', score)


# ------------------------------------------------
# score v01 ; RandomForestClassifier
# ------------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

model = RandomForestClassifier(n_estimators=500)
split_fit_score(X, y, model)   # 0.7662337662337663


# ------------------------------------------------
# eda & engineering
# ------------------------------------------------

# zero vals check
zero_cols = list(df.columns)[1:6]

print(df[df['Glucose'] == 0]['Glucose'].count(),
      df[df['Glucose'] != 0]['Glucose'].count())

for col in zero_cols:
    zero_cnt = df[df[col] == 0][col].count()
    zero_rate = zero_cnt / df.shape[0] * 100
    print(col, zero_cnt, round(zero_rate, 2))

print(df[df[zero_cols] > 0][zero_cols].describe())
print(df['Age'].describe())

df[df[zero_cols] > 0].hist(figsize=(12, 8))

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True)
# plt.show()   # insulin - glucose / bmi - skinthickness
             # outcome - glucoes, BMI, age


# binding ; age / 구간에 데이터가 없다면 구간을 넓게 설정
df['Age_band'] = pd.cut(df['Age'],
                        bins=[0, 20, 24, 29, 41, 82, 150],
                        labels=[0, 1, 2, 3, 4, 5])
print(df[['Age', 'Age_band']].head())
print(df['Age_band'].unique())


# outcome, age_band / bloodpressure
for col in zero_cols:
    print('-' * 30, col)
    print(df[df[col] > 0].groupby(['Outcome', 'Age_band'])[col].mean())


# zero to nan
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# fill zeros
# outcome, age_band ; insulin, bloodpressure, glucose
# outcome           ; BMI, skinthickness
out_age = ['Insulin', 'BloodPressure', 'Glucose']
out = ['BMI', 'SkinThickness']

df[out] = df[out].fillna(df.groupby('Outcome')[out].transform('mean'))
df[out_age] = df[out_age].fillna(df.groupby(['Outcome', 'Age_band'])[out_age].transform('mean'))

print(df.isna().sum())
df.hist(figsize=(12, 8))
# plt.show()


# ------------------------------------------------
# score v02 ; RandomForestClassifier
# ------------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = RandomForestClassifier(n_estimators=500)
split_fit_score(X, y, model)   # 0.8376623376623377


# ------------------------------------------------
# skew scaling
# outlier 제거 후, 정규화/간격조정/단위조정
# 1. MinMaxScaler   ; outlier에 민감
# 2. RobustScaler   ; 중앙 기준으로 쿼터 구분하여 조정
# 3. StandardScaler ; 모든 데이터를 평균 0, 편차 1로 조정
# tree model ; scale 영향 적음
# regression model ; scale 영향 큼
# y ; do only transfrom, fit not needed
# ------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scalers = [MinMaxScaler(), RobustScaler(), StandardScaler()]
model = RandomForestClassifier(n_estimators=500)

for i in scalers:
    i.fit(X)                      # scale
    scale_X = i.transform(X)      # apply scale
    print(i.__class__.__name__)   # print scale
    split_fit_score(scale_X, y, model)


# ------------------------------------------------
# score v03 ; scale, RandomForestClassifier
# ------------------------------------------------
# MinMaxScaler score   = 0.8376623376623377
# RobustScaler score   = 0.8506493506493507
# StandardScaler score = 0.8376623376623377


# ------------------------------------------------
# boost models
# !pip install xgboost
# !pip install lightgbm
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#
# wrapper class
# ------------------------------------------------
import xgboost as xgb
import lightgbm


# ------------------------------------------------
# score v04 ; scale, XGBClassifier
# 오버피팅을 제어할 수 있는 속성 포함
# 제어 속성 ; gamma=, reg_alpha=, reg_lambda=
# Gradient Descent(경사하강) ; convex 형태에서 loss를 최소화하는 최적점 찾기
# learning_rate=(보폭)에 따라 최적을 찾아갈 수 있음
# missing=np.nan ; 결측치 설정
# importance_type='gain' ; 정보량 우선 선택
# 가급적 param 튜닝 지양, default 사용
# ------------------------------------------------
model = xgb.XGBClassifier(n_estimators=300,
                          objective='binary:logistic') # =logloss
split_fit_score(scale_X, y, model)   # 0.8636363636363636


# ------------------------------------------------
# validation
# 검증, 증강 ; 데이터 N배 증강 후 순차적 학습
# 오버피팅 여부, score가 합당함을 보임
# 폴드의 개수만큼 데이터를 나눠서 학습하고 루프
# Fold            ; n spliting (KFold, StratifiedKFold
# cross_val_score ; n spliting & scoring n times
# GridSearchCV    ; n spliting & scoring n times & model tuning
# ------------------------------------------------
from sklearn.model_selection import KFold, StratifiedKFold

# KFold ; split return ndarray, 회귀 사용
kfold_score = []
kfold = KFold(n_splits=5, random_state=1414, shuffle=True)

for train_index, test_index in kfold.split(scale_X, y):

    sX_train, sX_test = scale_X[train_index], scale_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(sX_train, y_train)
    y_pred = model.predict(sX_test)
    score = accuracy_score(y_test, y_pred)
    print('kfold accuracy score = ', score)
    kfold_score.append(score)


# ------------------------------------------------
# score v05 ; scale, KFold split, XGBClassifier   # 0.8554367201426025
# ------------------------------------------------
print('kfold accuracy score mean = ', np.array(kfold_score).mean())


# StratifiedKFold ; y가 섞이지 않는 kfold를 보완하여 bootstrap, 회귀 불가, 분류 사용
skfold_score = []
skfold = StratifiedKFold(n_splits=5, random_state=1414, shuffle=True)

for train_index, test_index in kfold.split(scale_X, y):

    sX_train, sX_test = scale_X[train_index], scale_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(sX_train, y_train)
    y_pred = model.predict(sX_test)
    score = accuracy_score(y_test, y_pred)
    print('skfold accuracy score = ', score)
    skfold_score.append(score)


# ------------------------------------------------
# score v06 ; scale, StratifiedKFold split, XGBClassifier   # 0.8554367201426025
# ------------------------------------------------
print('skfold accuracy score mean = ', np.array(skfold_score).mean())


# cross_val_score
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(model, scale_X, y, scoring='accuracy', cv=5)
print(cv_score)


# ------------------------------------------------
# score v07 ; scale, cross_val_score, XGBClassifier   # 0.8698582463288347
# ------------------------------------------------
print('cv_score mean = ', cv_score.mean())