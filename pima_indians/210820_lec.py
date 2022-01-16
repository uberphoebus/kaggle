
# pima ---------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/02_pima/input/diabetes.csv')
print(df.info())

# --------------------------------------------------
# 요건이 갖춰지면 accuracy_score부터 확인
# 단계씩 가공 후 다시 accuracy_score 확인

# def ----------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

def dt_estimate(X, y, test_size=0.2, random_state=1414):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier(random_state=random_state)

    dt_model.fit(X_train, y_train)
    dt_prediction = dt_model.predict(X_test)

    from sklearn.metrics import accuracy_score
    dt_score = accuracy_score(y_test, dt_prediction)

    print('dt_score = ', dt_score)
def rf_estimate(X, y, test_size=0.2, random_state=1414, n_estimators=100):

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)


    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    rf_model.fit(X_train, y_train)
    rf_prediction = rf_model.predict(X_test)

    from sklearn.metrics import accuracy_score
    rf_score = accuracy_score(y_test, rf_prediction)

    print('rf_score = ', rf_score)

# estimate -----------------------------------------
rf_score = rf_estimate(X, y)
print('accuracy score v1 = ', rf_score)

# hist ---------------------------------------------
df.hist()
# plt.show()

# outlier ------------------------------------------
print(df.columns)

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print(df.shape)
print(df.head())

print(df['Glucose'][df['Glucose'] == 0].count())
print(df['Glucose'][df['Glucose'] != 0].count())

for i in zero_cols:
    print(i, df[df[i] == 0][i].count(), df[df[i] != 0][i].count())
    print(i, df[df[i] == 0][i].count() / df.shape[0], df[df[i] != 0][i].count() / df.shape[0])

print(df['BloodPressure'].mean())


# 나이 구간 별 혈압
# age ; min 21 ~ max 81
print(df.Age.describe())
# pd.cut(df['Age'], bins=[], labels=[])
# df['Age_band'] = df['Age'] // 10
# df['Age_band'] = df['Age'] // 20

# 구간 재설정; describe() 구간 0 초과 20 이하
df['Age_band'] = pd.cut(df['Age'], bins=[0, 20, 24, 29, 41, 82, 150], labels=[0, 1, 2, 3, 4, 5])

# print(df[['Age', 'Age_band']])
# print(df.groupby('Age_band')['BloodPressure'].mean())

# 당뇨 여부에 따른, 나이 구간별 혈압
# print(df.groupby(['Age_band', 'Outcome'])['BloodPressure'].mean())

# 혈당 수치에 따른, 나이 구간별 혈압
# print(df.groupby(['Age_band', 'Outcome'])['Glucose'].mean())

for i in zero_cols:
    print('-' * 40, i)
    print(df[df[i] > 0].groupby(['Age_band', 'Outcome'])[i].mean())

# 구간화를 세분화 했을 때 데이터가 없다면, 구간화를 넓게

# 상관관계
# outcome, age_band ; insulin, bloodpressure, glucose
# outcome           ; BMI, SkinThickness
out_age = ['Insulin', 'BloodPressure', 'Glucose']
only_out = ['BMI', 'SkinThickness']


# apply vs transform -------------------------------




# zero fill ----------------------------------------
# zero to nan
df[zero_cols] = df[zero_cols].replace(0, np.nan)

df['BMI'] = df['BMI'].fillna(df.groupby('Outcome')['Age'].transform('mean'))
df['SkinThickness'] = df['SkinThickness'].fillna(df.groupby('Outcome')['Age'].transform('mean'))

df[out_age] = df[out_age].fillna(df.groupby(['Age_band', 'Outcome'])['Age'].transform('mean'))

print(df.info())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# skew scaler --------------------------------------
# 정규화 및 간격을 좁히고 단위를 맞추기 위함
# outlier를 제거한 후 scale
# 1. MinMaxScaler ; outlier에 민감
# 2. RobustScaler ; 자주 사용. 중앙 기준으로 쿼터.
# 3. StandardScaler ; 모든 데이터를 평균 0, 편차 1로 맞춤

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
scalers = [MinMaxScaler(), RobustScaler(), StandardScaler()]

# for s in scalers

# scalers[0].fit() ; 20~80 값을 0과 1로 맞춤
# scalers[0].transform() ; fit 값들을 df에 적용
# fit, tran 나뉜 이유; 문제지 때문
# 문제지는 fit이 아니라, transform

# scalers[0].fit_transform() ; fit_transform


# scalers[0].fit(X)
# scaler_X = scalers[0].transform(X)

# print(scaler_X.reshape(-1, 1))

# rf_estimate(scaler_X, y)
# 이후에는 스케일링 된 문제지(X)를 넣어주면 됨
# 트리는 스케일링 영향을 잘 받지 않음, 회귀가 영향을 많이 받음

# for i in scalers:
#     i.fit(X)
#     scaler_X = i.transform(X)
#     print(i.__class__.__name__)
#     rf_estimate(scaler_X, y)

# boost model --------------------------------------
# !pip install xgboost
# !pip install lightgbm
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#
# 본래 c++로 만들어진 라이브러리
# wrapper class. c++ boost를 python으로 입출력하도록
# scikit-learn api
import xgboost as xgb
# import lighgbm

xg_model = xgb.XGBClassifier(n_estimators=300, objective='binary:logistic') # logloss
# 오버피팅을 제어할 수 있는 속성 포함
# gamma=, reg_lambda= 로 조정
# n_estimators=, max_depth=, learning_rate=경사하강(Gradient Descent)
# 경사하강; loss를 최소화하는 것을 찾음
# convex 모양에서 경사하강
# 경사하강의 거리에 따라 최적을 찾아갈수도 있고 지나칠수있음
# learning_rate = 보폭.
# reg_alpha=, reg_lambda= , 규제강도 조절
# missing=np.nan
# importance_type='gain' ; 정보량 (지니x)
# objective=

# split_fit_score()
# param 튜닝 지양, default 사용
# 공모전 분류모델 상위권은 boost 사용


# 증강/검증 ---------------------------------------------
# 모델 평가와 성능 향상
# skfold(straited), kfold
# **  cross-validation 교차검증; 매우 중요
# *** 그리드 서치; 매개변수 튜닝

# 교차검증 ; 폴드와 동일하나, 폴드를 단순화(점수만 받음)
# 폴드 ; 데이터를 n배 증강(증폭)시켜서 순차적 학습(과정, 점수)
# 그리드서치; 폴드와 동일하나, (과정, 점수, 모델 튜닝)

# 세 가지 검증 중 하나는 반드시 사용. 동시에 사용하는 것이 아님.
# 오버피팅인지 아닌지 확인이 필요함. 검증은 반드시 필요.
# 점수가 합당함을 보이는 과정

# https://www.clickai.ai/resource/wiki/modeling/crossvalidation_kor

# 한 데이터셋을 여럭개의 폴드로 나눔.
# 500개 라면 100개씩
# 5개의 폴드로 나눴다면, 1번을 문제 나머지를 학습
# 2번을 문제, 나머지 학습, ... 5번을 문제 나머지 학습
# 폴드의 개수만큼 루프를 돌림
# 500문제를 가지고, 2천 문제 학습 및 500문제 풀이
# 폴드 횟수는 무한이지만 5~10 정도

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score

kfold = KFold()
skfold = StratifiedKFold()

# 문제/답을 나눠서 넣을 필요 없이 문제 전부를 넣어야 함
# 점수들의 평균으로 평가
kfold.split(scale_X)

KFold(n_splits=5, shuffle=False) # 셔플하지 않음

for train_index, test_index in kfold.split(sclae_X):
    학습8, 검증2 = scale_X.iloc[train_index], scale_X.iloc[test_index]


skfold = StratifiedKFold(n_splits=5, shuffle=False)
for train_index, test_index in kfold.split(sclae_X, y):
    학습8, 검증2 = scale_X.iloc[train_index], scale_X.iloc[test_index]
    학습답8, 검증답2 = y.iloc[train_index], y.iloc[test_index]

# 학습할 때 이것부터 해도 됨
# 피쳐를 바꿀 때마다 한참 기다려야 함
# 처리 속도가 오래 걸림림
# 폴드의 단점. 답안지가 골고루 섞이지 않을 수 있음
# skfold ; 학습을 잘 할 수 있도록 답안 섞음 bootstrap
# 폴드는 골고루 넣을 수 없는 데이터에서 사용, 회귀에서는 골고루가 안됨.
# 분류에서만 골고루 가능
# 회귀는 무조건 kfold만 사용
# 분류는 skfold
# kfold ; train_test_split()
# skfold ; train_test_split(test_size=0.2, stratify=y)

# cross-val; 점수만 받음
total_score = cross_val_score(xg_model, scale_X, y, scoring='accuracy', cv=5)
# cv = n_splits
# skl 내 estimator만 사용 가능.
