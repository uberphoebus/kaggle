
# ------------------------------------------------
# sklearn 분류 성능 평가 classification_report
# ------------------------------------------------

# ------------------------------------------------
# sklearn.metrics confusion_matrix
# index=iy_true, cols=iy_pred
# loss 분석 가능, 가장 많이 사용, 오차행렬
# confusion_matrix
#           TN | FP
#           -------
#           FN | TP
# accuracy_score(정확도)  = (TP + TN) / (TP + FP + TN + FN)
# precision_score(정밀도) =  TP / (TP + FP)
# recall_score(재현율)    = TP / (FN + FP)
# f1_score(조화평균)      = 2 * (precision * recall) / (precision + recall)
# ------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_boston

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve


# ------------------------------------------------
# iris scores
# ------------------------------------------------

iris = load_iris()
iris_df = pd.DataFrame(iris['data'])
iris_df['target'] = iris['target']
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

# train_test_split
iX = iris_df.drop('target', axis=1)
iy = iris_df['target']
iX_train, iX_val, iy_train, iy_val = train_test_split(iX, iy, train_size=0.2, random_state=1414)

# model fit, pred, score
model = RandomForestClassifier(random_state=1414)
model.fit(iX_train, iy_train)
iy_pred = model.predict(iX_val)

def my_eval(y_val, y_pred, average='binary'):

    """accuracy_score, precision_score, recall_score, f1_score"""

    print('-- my_eval', '-' * 40)

    acc_score = accuracy_score  (y_val, y_pred)
    pre_score = precision_score (y_val, y_pred, average=average)
    rec_score = recall_score    (y_val, y_pred, average=average)
    f_score   = f1_score        (y_val, y_pred, average=average)

    print('-' * 50)
    print('accuracy_score  = {:.4f}'.format(acc_score))
    print('precision_score = {:.4f}'.format(pre_score))
    print('recall_score    = {:.4f}'.format(rec_score))
    print('f1_score        = {:.4f}'.format(f_score))
    print('정확도 = {:.4f}, 정밀도 = {:.4f}, 재현율 = {:.4f}, f1 = {:.4f}'
          .format(acc_score, pre_score, rec_score, f_score))
    print(f'정확도 = {acc_score:.4f}, 정밀도 = {pre_score:.4f}, '
          f'재현율 = {rec_score:.4f}, f1 = {f_score:.4f}')
    print('-' * 50)

    print('confusion_matrix\n', confusion_matrix(y_val, y_pred))
    # print(classification_report(y_val, y_pred))   # 근사치
    print('-' * 50)

my_eval(iy_val, iy_pred, average='macro')


# ------------------------------------------------
# diabetes scores
# ------------------------------------------------

diabetes = pd.read_csv('../kaggle/pima/input/diabetes.csv')
dX = diabetes.drop('Outcome', axis=1)
dy = diabetes['Outcome']
dX_train, dX_val, dy_train, dy_val = train_test_split(dX, dy, train_size=0.2, random_state=1414, shuffle=True, stratify=dy)
model = RandomForestClassifier(random_state=1414)
model.fit(dX_train, dy_train)
dy_pred = model.predict(dX_val)

my_eval(dy_val, dy_pred)