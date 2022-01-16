# __init__------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split # def
from sklearn.tree import DecisionTreeClassifier      # class
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score           # def

import warnings
warnings.filterwarnings(action='ignore') # 경고 무시 옵션

train = pd.read_csv('./kaggle/01_titanic/input/train.csv')
test = pd.read_csv('./kaggle/01_titanic/input/test.csv')
submission = pd.read_csv('./kaggle/01_titanic/input/gender_submission.csv')

def nan_check():
    """missing val check frame"""
    nan_dict = {'CNT':train_test.isna().sum(),
                'RATE':train_test.isna().sum()
                       / train_test.shape[0] * 100}
    nan_df = pd.DataFrame(nan_dict)
    return nan_df.head().T
# --------------------------------------------------
train_test = train.append(test)
train_test['Cabin'].fillna('U0', inplace=True) # return 없음
# --------------------------------------------------

# --------------------------------------------------
# encoding
# 1. .replace()   ; 단순 문자열 변경
# 2. .map()
# 3. .apply(lambda)
# 4. .transform()
# --------------------------------------------------

# .apply(lambda) ; Sex col
train_test['Sex'] = train_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# --------------------------------------------------
# srs.unique()       # distinct
# srs.nunique()      # count distinct
# srs.value_counts()
# srs.mode()         # 최빈값
# --------------------------------------------------
em_uni  = train_test['Embarked'].unique()
em_nuni = train_test['Embarked'].nunique()
em_mode = train_test['Embarked'].mode()
print('uni = ', em_uni, 'nuni = ', em_nuni, 'mode = ', em_mode)

# .apply(lambda) ; Embarked col 3개 값
train_test['Embarked'] = train_test['Embarked'].fillna('S')
print('em isna = ', train_test['Embarked'].isna().sum())
train_test['Embarked'] = train_test['Embarked'].apply(lambda x: 0 if x == 'S'
                                                      else (1 if x == 'C' else 2))
print(train_test['Embarked'].unique())

# --------------------------------------------------
# 정규표현식(regex)
# a-z, A-Z, 0-9
# ? ; 0 또는 1 번 발생 ab?c ac abc
# * ; 0 번 이상       ab*c ac abc abbc
# + ; 1 번 이상       ab+c abc abbc
# {} 구문반복, / 검색 시작, . 1개 문자, [] 문자 클래스
# ^ 처음, $ 끝, | 또는, [ ] 공백
# --------------------------------------------------

# .map() ; Name_title, .extract()
# ([A-Za-z]+)\.
train_test['Name_title'] = train_test['Name'].str.extract('([A-Za-z]+)\.')
print(train_test.Name_title.value_counts())   # 호칭별 통계

title_map = {'Mr':0,
             'Miss':1, 'Mlle':1, 'Ms':1, 'Lady':1,
             'Mrs':2, 'Mme':2,
             'Master':3, 'Dr':3, 'Sir':3,
             'Rev':4, 'Col':4, 'Major':4, 'Don':4,
             'Capt':4, 'Countess':4, 'Jonkheer':4, 'Dona':4}
train_test['Name_title'] = train_test['Name_title'].map(title_map)

# .transform() ; age null에  호칭별 평균 나이 부여
age_gr = train_test.groupby('Name_title')['Age']
print(age_gr.mean())
train_test['Age'] = train_test['Age'].fillna(age_gr.transform('mean'))
print('age null = ', train_test['Age'].isna().sum())
train_test['Age'] = np.round(train_test['Age'], 0).astype('int64')

# family 파생변수 ; Fare 결측치 처리
train_test['family'] = train_test['SibSp'] + train_test['Parch'] + 1
print(train_test[train_test['Fare'].isna()])   # fare null
fam_pc_cond = train_test[(train_test['family'] == 1) & (train_test['Pclass'] == 3)]
mean_fare = fam_pc_cond['Fare'].mean()   # 조건의 평균
train_test['Fare'] = train_test['Fare'].fillna(mean_fare)
mean_fare = round(mean_fare, 4)
print(train_test['Fare'].isna().sum())


# --------------------------------------------------
# binding ; age
# --------------------------------------------------
sns.set()   # 배경 설정
sns.histplot(data=train, x='Age', hue='Survived')
# print(plt.show())  # hist check

# 1. .loc[] ; Age_band1
train_test.loc[train_test['Age'] <= 16 , 'Age_band1'] = 1
train_test.loc[(train_test['Age'] > 16) & (train_test['Age'] <= 34), 'Age_band1'] = 2
train_test.loc[(train_test['Age'] > 34) & (train_test['Age'] <= 48), 'Age_band1'] = 3
train_test.loc[(train_test['Age'] > 48) & (train_test['Age'] <= 67), 'Age_band1'] = 4
train_test.loc[train_test['Age'] > 67 , 'Age_band1'] = 5
train_test['Age_band1'] = train_test['Age_band1'].astype('int64')

# 2. pd.cut(df, bins=[], labels=[])
train_test['Age_band2'] = pd.cut(train_test['Age'],
                                 bins=[0, 16, 34, 48, 67, 90],
                                 labels=[1, 2, 3, 4, 5])
# train_test['Age_band2'] = train_test['Age_band2'].astype('int64')

# 3. 단순 연산
train_test['Age_band3'] = train_test['Age'] // 10

# --------------------------------------------------
# drop cols
# --------------------------------------------------
drop_cols = ['Name', 'Age', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Age_band2', 'Age_band3']
train_test.drop(drop_cols, axis=1, inplace=True)


# --------------------------------------------------
# 학습 모델 선정
# https://scikit-learn.org/stable/
# 회귀(regression)모델; linear
# 1.1. Linear Models
# 1.2. Linear and Quadratic Discriminant Analysis
# 1.3. Kernel ridge regression
# 1.4. Support Vector Machines     ; 차원 축소
# 1.5. Stochastic Gradient Descent ; loss 관련 부분
# 1.6. Nearest Neighbors           ; 이웃노드, 비슷한 유형, 분류
# 1.7. Gaussian Processes          ; 정규분포
# 1.8. Cross decomposition
# 1.9. Naive Bayes                 ; 이진(바이너리) 분류
# 1.10. Decision Trees             ; 트리 형태
# 1.11. Ensemble methods           ; 앙상블, 집단지성? 여러 모델의 결과를 도합하여 도출
# 1.12. Multiclass and multioutput algorithms ; 다중 분류
# 1.13. Feature selection

# sklearn.tree.DecisionTreeClassifier
# Discrete versus Real AdaBoost       ; 조건 = 빅데이터

# 모델 학습 절차
# *train_test_split()
# 1. 학습 [*fit()],     train
# 2. 예측 [*predict()], validation
# 3. 평가 [*score()],   test ; overfitting/underfitting
# 4. 검증
# 5. 튜닝      ; 분석가 영역 XGBoost 40~80
# --------------------------------------------------

# --------------------------------------------------
# train - test 분리 ; 모델 학습 전 분리
# --------------------------------------------------
print(train.shape[0], test.shape[0])
train = train_test[~train_test['Survived'].isna()]
test  = train_test[train_test['Survived'].isna()]

train['Survived'] = train['Survived'].astype('int64')
test.drop('Survived', axis=1, inplace=True)
print(train.info(), test.info())
print('train, test shape = ',train.shape, test.shape)

# --------------------------------------------------
# .train_tset_split()
# .fit()      ; 학습 train 8
# .prediect() ; 예측 train 2 validation
# .score()    ; 평가
# skl = sklearn.model_selection.\
#     train_test_split(arrays,            # data
#                      test_size=None,    # 0.2
#                      train_size=None,   # 0.8
#                      random_state=None, # 특정 난수표 선택
#                      shuffle=True,      # 시계열일 때 False
#                      stratify=None)     # train-test balance
# --------------------------------------------------
from sklearn.model_selection import train_test_split # def

X = train.drop('Survived', axis=1) # 전체 문제
y = train['Survived']              # 전체 답지
print('X, y shape = ',X.shape, y.shape)

# _train : _val = 8 : 2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# --------------------------------------------------
# sklearn.tree.DecisionTreeClassifier model
# --------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)     # .fit() ; 기계학습
dt_pred = dt_model.predict(X_val)  # .predict() ; 예측

from sklearn.metrics import accuracy_score
dt_acc_score = accuracy_score(y_val, dt_pred) # 자가평가/검증
print('dt model score = ', dt_acc_score)

# --------------------------------------------------
# sklearn.ensemble.RandomForestClassifier model
# --------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)     # .fit() ; 기계학습
rf_pred = rf_model.predict(X_val)  # .predict() ; 예측

rf_acc_score = accuracy_score(y_val, rf_pred) # 자가평가/검증
print('rf model score = ', rf_acc_score)


# --------------------------------------------------
# submit
# --------------------------------------------------
rf_pred = rf_model.predict(test)  # array
print(type(rf_pred), submission.shape, test.shape)

submission['Survived'] = rf_pred.reshape(-1, 1)
print(submission.head(10))

submission.to_csv('./kaggle/01_titanic/titanic_sub_v01.csv', index=False)
df = pd.read_csv('./kaggle/01_titanic/titanic_sub_v01.csv')
print(df.head(10))

# --------------------------------------------------
# .feature_importances_
# --------------------------------------------------
rf_fi = rf_model.feature_importances_ * 100
rf_cols = test.columns.values
df = pd.DataFrame(rf_fi, index=rf_cols)
df = df.sort_values(by=0, ascending=False)
print(df)

dict = {'feature':rf_cols, 'imp':rf_fi}
impdf = pd.DataFrame(dict).sort_values(by='imp', ascending=False).T
print(impdf)