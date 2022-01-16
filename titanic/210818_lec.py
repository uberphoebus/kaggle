
# __init__------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
print(sklearn.__version__)

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

drop_cols = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
# train_test.drop(drop_cols, axis=1, inplace=True)
# --------------------------------------------------
# 1. train, validation 결합
# 2. 대략적 시각화
# 3. target label balance
# 4. missing val
# 5. encoding
# 6. 파생변수 설정
# 7. binding(구간화)
# 8. 표준화
# 9. 스케일링
# 10. outlier 처리
# --------------------------------------------------
train_test = train.append(test)
print(train_test.shape), print(train_test.info())

train_test['Cabin'].fillna('U0', inplace=True) # return 없음

train_test['Cabin'] = train_test['Cabin'].str[0]
train['Cabin'] = train['Cabin'].str[0]
cross_s = pd.crosstab(train['Cabin'], train['Survived'])
cross_p = pd.crosstab(train['Cabin'], train['Pclass'])
print(cross_s), print(cross_p)

# encoding; .replace(), .map(), .apply(lambda)
print(train_test['Sex'].head())
cp = train_test['Sex'].head()
cp = cp.replace('male', 1)
cp = cp.replace('female', 0)
print(cp)

# .map(), .apply() 구분; 단일 컬럼이면 map, 다중 컬럼이면 apply
# .map() ; replace보다 빠름
cp2 = train_test[['Sex', 'Age', 'Embarked']].head()
dict = {'male':0, 'female':1}        # mapping_dict
cp2['Sex2'] = cp2['Sex'].map(dict)   # df[cl].map(mapping_dict)
print(cp2.head())

cp2['Sex3'] = cp2['Sex'].apply(lambda x: 0 if x == 'male' else 1)

# distincts, 최빈값 확인
print(train_test['Embarked'].unique())    # sql의 distinct
print(train_test['Embarked'].nunique())   # sql의 distinct 개수
print(train_test['Embarked'].value_counts())
print(train_test['Embarked'].mode())      # 최빈값

# 3개 이상 lambda
cp2['Embarked2'] = cp2['Embarked'].apply(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))
print(cp2)

# encoding; Sex
train_test['Sex'] = train_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# encoding; Embarked
emb_map = {'S':0, 'C':1, 'Q':2}
train_test['Embarked'].fillna('S', inplace=True)
print(train_test['Embarked'].isna().sum())
train_test['Embarked'] = train_test['Embarked'].map(emb_map)
print(train_test[['Sex', 'Embarked']].head())
print(train_test['Embarked'].unique())

# --------------------------------------------------
# 정규표현식(regex)
# abcd@efg.com
# a-zA-Z0-9@a-zA-Z.a-zA-Z
# ? ; 0 또는 1 번 발생 ab?c ac abc
# * ; 0 번 이상       ab*c ac abc abbc
# + ; 1 번 이상       ab+c abc abbc
# {}; 구분 반복       {a-z} ajsdfoijsozadifjiaoz
# / ; 검색 시작 명령
# . ; 1개 문자
# []; 문자 클래스
# ^ ; 처음
# $ ; 끝
# | ; 또는
# [ ]; 공백
# --------------------------------------------------
print(train_test['Name'].head())
# ([A-Za-z]+)\.
train_test['Name_title'] = train_test['Name'].str.extract('([A-Za-z]+)\.')
print(train_test['Name_title'].value_counts())

print(train_test['Age'].isna().sum())
print(train_test[train_test['Age'].isna()]['Name_title'])
print(train_test[train_test['Age'].isna()]['Name_title'].unique())

title_map = {'Mr':0,
             'Miss':1, 'Mlle':1, 'Ms':1, 'Lady':1,
             'Mrs':2, 'Mme':2,
             'Master':3, 'Dr':3, 'Sir':3,
             'Rev':4, 'Col':4, 'Major':4, 'Don':4,
             'Capt':4, 'Countess':4, 'Jonkheer':4, 'Dona':4}
train_test['Name_title'] = train_test['Name_title'].map(title_map)
print(train_test[['Name', 'Name_title']].head())


print(train_test.groupby(by='Name_title').mean()['Age'])
print(train_test.groupby(by='Name_title')['Age'].mean())

# transform

train_test['Age'] = train_test['Age'].fillna(train_test.groupby(by='Name_title')['Age'].transform('mean'))
print('after transform Age na = ', train_test['Age'].isna().sum())

# 분석 전 시나리오 타이핑을 해야 나중에 처리했는지 헷갈리지 않음

# family 파생변수 처리
train_test['family'] = train_test['SibSp'] + train_test['Parch'] + 1
print(train_test[['family', 'SibSp', 'Parch', 'Pclass', 'Fare']].head(20))
# pclass에 따라 요금 차이 확인, 같은 등급이어도 또 다름

# fare null
print(train_test[train_test['Fare'].isna()][['family', 'SibSp', 'Parch', 'Pclass', 'Fare']])

# 혼자 탄 사람 중 3등급
print('befor fare null = ', train_test['Fare'].isna().sum())
cond = train_test[(train_test['family'] == 1) & (train_test['Pclass'] == 3)]
mean_fare = cond['Fare'].mean()
# mean_fare2 = cond['Fare'].transform('mean')   # transform check
mean_fare = round(mean_fare, 4)
train_test['Fare'] = train_test['Fare'].fillna(mean_fare)
print('after fare null = ', train_test['Fare'].isna().sum())

print(train_test.isna().sum())          # 결측치 중간 점검
print(train_test.info()) # drop cols check

# cols drop
drop_cols = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
train_test = train_test.drop(drop_cols, axis=1)
print(train_test.info())


# plot
sns.set()
sns.histplot(data=train, x='Age', hue='Survived')
# print(plt.show())


# 구간화
train_test['Age'] = np.round(train['Age'], 0).astype('int64')

train_test.loc[train_test['Age'] <= 16, 'Age_band'] = 1
train_test.loc[(train_test['Age'] >= 16) & (train_test['Age'] < 35), 'Age_band'] = 2
train_test.loc[(train_test['Age'] >= 35) & (train_test['Age'] < 48), 'Age_band'] = 3
train_test.loc[(train_test['Age'] >= 48) & (train_test['Age'] < 67), 'Age_band'] = 4
train_test.loc[(train_test['Age'] >= 67), 'Age_band'] = 5

train_test['Age_band'] = train_test['Age_band'].astype('int64')
print(train_test[['Age', 'Age_band']].head())

train_test.drop('Age', axis=1, inplace=True)


# 구간 나누기
# pd.cut(cols, n); 동일 길이
# pd.cut(cols, bins=[0, 16, 34, 48, 67, 90], labels=[1, 2, 3, 4, 5]); 동일 길이
# pd.qcut(cols, n); 동일 개수

# age_band3
# train_test['Age_band3'] = train_test['Age'] // 10




# frame 분리
print(train.shape[0], test.shape[0])
print(train_test['Survived'].isna().sum())

train = train_test[~train_test['Survived'].isna()]
test  = train_test[ train_test['Survived'].isna()]
print(train.info(), test.info())
print(train.shape, test.shape)

train2 = train_test.iloc[:train.shape[0] ]
test2 = train_test.iloc[ train.shape[0]:]
print(train2.shape, test2.shape)

test = test.drop('Survived', axis=1)
print(test.shape)

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

from sklearn.tree import DecisionTreeClassifier      # class
from sklearn.metrics import accuracy_score           # def
from sklearn.model_selection import train_test_split # def

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test#sklearn.model_selection.train_test_split
# arrays = np.arange(10)
# sk_test = sklearn.model_selection.train_test_split(arrays,           # data
#                                                    test_size=None,    # 0.2
#                                                    train_size=None,   # 0.8
#                                                    random_state=None, # 특정 난수표 설정
#                                                    shuffle=True,      # 시계열일때는 False
#                                                    stratify=None)     # train_test balance

import warnings
warnings.filterwarnings(action='ignore') # 경고 무시 옵션
train['Survived'] = train['Survived'].astype('int64').copy()
print(train.info())


# X 문제지, y 답안지
y = train['Survived']
X = train.drop('Survived', axis=1)
print(X.shape, y.shape)
print(X.head())
print(y.head())

dt_model = DecisionTreeClassifier()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

dt_model.fit(X_train, y_train)   # 기계학습
pred = dt_model.predict(X_val)   # 예측 답안

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_val, pred)
print('dt 모델 점수 = ', acc_score)

# RandomForest
from sklearn.ensemble import RandomForestClassifier
dt_model = RandomForestClassifier()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

dt_model.fit(X_train, y_train)   # 기계학습

# 셀프 테스트
pred = dt_model.predict(X_val)   # 예측 답안
acc_score = accuracy_score(y_val, pred)
print('rf 모델 점수 = ', acc_score)

# 리더보드 제출은 real_pred
real_pred = dt_model.predict(test)
print(real_pred[:20])

# 공모전 답안지 제출
print(submission.shape)
print(test.shape)

submission['Survived'] = real_pred.reshape(-1, 1)
print(submission.head(10))

submission.to_csv('./kaggle/01_titanic/titanic_sub_v01.csv', index=False)
df = pd.read_csv('./kaggle/01_titanic/titanic_sub_v01.csv')
print(df.head(10))

# 개선 시도; binding 구간화
print(train.head(10))

# feature별 중요도
print(dt_model.feature_importances_ * 100)
print(test.columns.values)
features = test.columns.values
f_scores = dt_model.feature_importances_ * 100
df = pd.DataFrame(f_scores, index=features)
print(df.sort_values(by=0, ascending=False))

# dict 결합
dict = {'feature':test.columns.values,
        'imp':dt_model.feature_importances_}
impdf = pd.DataFrame(dict)
impdf = impdf.sort_values(by='imp', ascending=False).T
print(impdf)

# kaggle 당뇨병