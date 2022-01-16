
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# ------------------------------------------------
# iris
# ------------------------------------------------

iris = load_iris()    # arr type
print(iris.keys())
print(iris['target_names'])
print(iris['feature_names'])
# 꽃잎의 넓이/길이, 꽃받침의 넓이/길이에 따라 구분

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df = pd.DataFrame(data=iris['data'])
print(iris_df.info())

# iris_df + target
iris_df['target'] = iris['target']
s = pd.Series(iris['target'], name='target1')
iris_df = pd.concat([iris_df, s], axis=1); print(iris_df.info())
iris_df.drop('target1', axis=1, inplace=True); print(iris_df.info())

# rename cols
col_names = ['sepal_length','sepal_width','petal_length','petal_width']
iris_df.columns = col_names + ['target0']; print(iris_df.info())
iris_df.rename(columns={'target0':'target'}, inplace=True); print(iris_df.info())

# index check
print(iris_df.index.values[:10])

# target data check
print(iris_df['target'].unique())  # df[col].unique()
print(np.unique(iris['target']))   # np.unique(arr)
train_test_split()
# 1. X, y sep
# 2. train_test sep
# 3. model selection
# 4. model.fit()
# 5. model.predict()
# 6. accuracy_score()

X = iris_df.drop('target', axis=1)
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=10, shuffle=True, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ------------------------------------------------
# iris fold
# ------------------------------------------------
# fold ; train_test를 나누지 않을 것이기 때문에 전체 data 입력
kfold = KFold(n_splits=3, shuffle=True, random_state=555)

tot_score = []
for train_index, test_index in kfold.split(X): # 문제만 필요, return arr type index
    X_test, X_train = X.iloc[test_index], X.iloc[train_index]  # scaling 경우 iloc 없음
    y_test, y_train = y.iloc[test_index], y.iloc[train_index]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    tot_score.append(score)
print('iris fold score = ', np.array(tot_score).mean())

# ------------------------------------------------
# credit card fold
# ------------------------------------------------
credit = pd.read_csv('./kaggle/creditcard/UCI_Credit_Card.csv')

c_X = credit.drop('default.payment.next.month', axis=1)
c_y = credit['default.payment.next.month']

c_kfold = KFold(n_splits=5, shuffle=True, random_state=555)

# ctot_score = []
# for train_index, test_index in c_kfold.split(c_X):
#     cX_train, cX_test = c_X.iloc[train_index], c_X.iloc[test_index]
#     cy_train, cy_test = c_y.iloc[train_index], c_y.iloc[test_index]
#
#     c_model = RandomForestClassifier()
#     c_model.fit(cX_train, cy_train)
#     cy_pred = c_model.predict(cX_test)
#     c_score = accuracy_score(cy_test, cy_pred)
#     ctot_score.append(c_score)
# print('credit fold score = ',np.array(ctot_score).mean())


# ------------------------------------------------
# iris skfold
# ------------------------------------------------
# target의 전체 label 비율과 각 fold의 label 비율을 동일하도록
# 반드시 label이 있는 y 필요
skfold = StratifiedKFold(n_splits=3, shuffle=False) # , random_state=555)
skfold.split(X, y) # 문제/정답 필요

sktot_score = []
for train_index, test_index in skfold.split(X, y):
    skX_train, skX_test = X.iloc[train_index], X.iloc[test_index]
    sky_train, sky_test = y.iloc[train_index], y.iloc[test_index]
    # print(sky_test.value_counts())

    sk_model = RandomForestClassifier()
    sk_model.fit(skX_train, sky_train)
    sky_pred = sk_model.predict(skX_test)
    sk_score = accuracy_score(sky_test, sky_pred)
    sktot_score.append(sk_score)
print('iris skfold score = ', np.array(sktot_score).mean())


# ------------------------------------------------
# iris cross_val_score
# ------------------------------------------------

# cross_val_score(estimator,     # model
#                 X, y=None, *,
#                 scoring=None,  # 'accuracy'
#                 cv=None)       # n_splits= / retrun ndarray scores

iris_cvs = cross_val_score(model, X, y, scoring='accuracy', cv=3)
print('iris cross_val_score = ', iris_cvs.mean())


# ------------------------------------------------
# credit card cross_val_score
# ------------------------------------------------
# model = RandomForestClassifier()   # model에서 shuffle, random_state 설정
# credit_cvs = cross_val_score(model, c_X, c_y, scoring='accuracy', cv=5)
# # print(credit_cvs)
# print('credit cross_val_score = ', credit_cvs.mean())


# ------------------------------------------------
# GridSearchCV ; kfold + cvs + tuning
# ------------------------------------------------
from sklearn.model_selection import GridSearchCV

# gcv = GridSearchCV(estimator,    # model ; list 가능
#                    param_grid,   # hyper param ; model에 들어가는 param을 수정
#                    scoring=None, # 'accuracy'
#                    refit=True,
#                    cv=None,      # kfold, skfold 입력 가능
#                    return_train_score=False)

# model = RandomForestClassifier(n_estimators=100,    # tree count
#                                max_depth=4,         # tree depth
#                                min_samples_split=2) # 나눌 샘플의 최소 개수

hyper_param = {'n_estimators':[100, 200, 300, 400],
               'max_depth':[5, 6, 7, 8],
               'min_samples_split':[1, 2, 3, 4, 5]}
# 4 * 4 * 5 = 80번 / 80 * 3(fold) = 240회 / 240회 * 400
# 가장 좋은 param을 반영, refit=, 다 테스트 하고 좋은 파라미터를 넣어두는 설정

refit_model = GridSearchCV(model, param_grid=hyper_param,
                           scoring='accuracy', refit=True,
                           cv=3, return_train_score=True)
# refit_model.fit(X, y)
# print(refit_model.best_params_)
# print(refit_model.best_score_)
# print(refit_model.best_estimator_)

# pred = refit_model.predict(test)


# ------------------------------------------------
# credit card GridSearchCV
# ------------------------------------------------

model = RandomForestClassifier()
hyper_param = {'n_estimators':[100, 200, 300, 400, 500],
               'max_depth':[5, 6, 7, 8, 9],
               'min_samples_split':[1, 2, 3, 4, 5]}
credit_skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=555)
credit_gscv = GridSearchCV(model, param_grid=hyper_param, scoring='accuracy', refit=True, cv=credit_skfold)

credit_gscv.fit(c_X, c_y)

print('credit_gscv best params    = ', credit_gscv.best_params_)
print('credit_gscv best score     = ', credit_gscv.best_score_)
print('credit_gscv best estimator = ', credit_gscv.best_estimator_)

# graphviz 는 tree 만 확인 가능