
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.datasets import load_iris, load_boston

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


# ------------------------------------------------
# iris
# ------------------------------------------------
iris = load_iris()
iris_df = pd.DataFrame(iris['data'])
iris_df['target'] = iris['target']
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
print(iris_df.info())
print(iris_df['target'].unique())   # df[col].unique()
X = iris_df.drop('target', axis=1)
y = iris_df['target']

# train_test_split + RandomForestClassifier + accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=555, shuffle=True, stratify=y)
model = RandomForestClassifier(random_state=555)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('iris tts_rf_acc score = ', accuracy_score(y_test, y_pred))

# KFold + RandomForestClassifier + accuracy_score
kfold = KFold(n_splits=3, shuffle=True, random_state=555)
k_score = []
for train_index, test_index in kfold.split(X):   # return arr type index
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]   # scaling -> arr
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier(random_state=555)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    k_score.append(accuracy_score(y_test, y_pred))
print('iris kfold_rf_acc score = ', np.array(k_score).mean())

# StratifiedKFold + RandomForestClassifier + accuracy_score
# target의 label비율과 각 fold의 label 비율 동일하도록
skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=555)
sk_score = []
for train_index, test_index in skfold.split(X, y): # target label 비율 필요
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier(random_state=555)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    sk_score.append(accuracy_score(y_test, y_pred))
print('iris skfold_rf_acc score = ', np.array(sk_score).mean())

# cross_val_score + RandomForestClassifier
# cross_val_score(estimator,     # model
#                 X, y=None, *,
#                 scoring=None,  # 'accuracy'
#                 cv=None)       # n_splits= / retrun ndarray scores
model = RandomForestClassifier(random_state=555)
cv_score = cross_val_score(model, X, y, scoring='accuracy', cv=3)
print('iris rf cross_val_score = ', cv_score.mean())

# StratifiedKFold + RandomForestClassifier + GridSearchCV
# gcv = GridSearchCV(estimator,               # model
#                    param_grid,              # hyper param ; model에 들어가는 param을 수정
#                    scoring=None,            # 'accuracy'
#                    refit=True,              # 점수에 따라 결과 param 조정하여 적용
#                    cv=None,                 # int, kfold, skfold 입력 가능
#                    return_train_score=True) # scoring
skfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=555)
model = RandomForestClassifier(random_state=555)
hyper_param = {'n_estimators'     :[100, 200, 300],
               'max_depth'        :[5, 6, 7, 8],
               'min_samples_split':[2, 3, 4, 5]}
gscv = GridSearchCV(model, param_grid=hyper_param, scoring='accuracy',
                    refit=True, cv=skfold, return_train_score=True)
gscv.fit(X, y)

print('iris sk_rf_gscv best score     = ', gscv.best_score_)
print('iris sk_rf_gscv best index     = ', gscv.best_index_)
print('iris sk_rf_gscv best params    = ', gscv.best_params_)
print('iris sk_rf_gscv best estimator = ', gscv.best_estimator_)
# print('iris sk_rf_gscv cv_results     = ', gscv.cv_results_)