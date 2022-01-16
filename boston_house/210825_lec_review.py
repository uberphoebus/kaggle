
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_boston

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

# ------------------------------------------------
# sklearn classification metrics & scoring ; 분류 성능 평가
# ------------------------------------------------

iris = load_iris()
iris_df = pd.DataFrame(iris['data'])
iris_df['target'] = iris['target']
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

X = iris_df.drop('target', axis=1)
y = iris_df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.2, random_state=1414)

model = RandomForestClassifier(random_state=1414)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# ------------------------------------------------
# confusion_matrix = TN, FP, FN, TP
# accuracy_score   = (TP + TN) / (TP + FP + FN + TP)
# precision_score  = TP / (FP + TP)
# recall_score     = TP / (FN + TP)
# f1_score         = 2 * (precision * recall) / (precision + recall)
# ------------------------------------------------

# classification metrics & scoring func
def my_eval(y_val, y_pred, average='macro'):

    """accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report"""

    print('*' * 20, 'my_eval ', '*' * 20)

    acc_score = accuracy_score  (y_val, y_pred)
    pre_score = precision_score (y_val, y_pred, average=average)
    rec_score = recall_score    (y_val, y_pred, average=average)
    f_score   = f1_score        (y_val, y_pred, average=average)

    print('-' * 50)
    print(f'accuracy_score  = {acc_score:.4f}\n'
          f'precision_score = {pre_score:.4f}\n'
          f'recall_score    = {rec_score:.4f}\n'
          f'f1_score        = {f_score:.4f}')

    print('-' * 50)
    print('confusion_matrix\n', confusion_matrix(y_val, y_pred))

    print('-' * 50)
    print('classification_report\n', classification_report(y_val, y_pred)) # 근사치

my_eval(y_val, y_pred)


# diabetes example

boston = load_boston()
boston_df = pd.DataFrame(boston['data'])
boston_df['target'] = boston['target']
boston_df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target']

X = boston_df.drop('target', axis=1)
y = boston_df['target']

print(X.info(), y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.2, random_state=1414)

model = RandomForestClassifier(random_state=1414)
model.fit(X_train, y_train)
# y_pred = model.predict(X_val)   # error
# my_eval(y_val, y_pred, average='macro')
