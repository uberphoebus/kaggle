# sklearn dataset : https://scikit-learn.org/stable/datasets/index.html

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold


def SCORES(y_val, pred, proba, str=None, cls_type=None) :
    if cls_type == "m" :
        # print("===========Multi Classifier======")
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average='macro')
        auc = roc_auc_score(y_val, proba, average='macro', multi_class='ovo')
        print('{} acc {:.4f}  f1 {:.4f}  auc {:.4f}'.format(str, acc, f1, auc))
    else :
        # print("===========Binary Classifier======")
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        auc = roc_auc_score(y_val, proba[:,1])
        print('acc {:.4f}  f1 {:.4f}  auc {:.4f}  {}'.format(acc, f1, auc, str))


# dataset = load_iris()
# df = pd.DataFrame(data=dataset.data,
#                   #columns=dataset.feature_names
#                   columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#                   )
# clstype = "m"

dataset = load_breast_cancer()
df = pd.DataFrame(data=dataset.data,
    columns=dataset.feature_names
)
clstype = "s"



df["target"] = dataset.target
X_train = df.iloc[: , :-1]
y_train = df.iloc[: , -1]

X_train7 , X_val3, y_train7, y_val3 = train_test_split(X_train, y_train, test_size=0.2, random_state=121)

rf_model = RandomForestClassifier()
rf_model.fit(X_train7, y_train7)
rf_pred3 = rf_model.predict(X_val3)
rf_proba3 = rf_model.predict_proba(X_val3)
SCORES(y_val3, rf_pred3, rf_proba3, "[RandomForestClassifier] ", cls_type=clstype)

#----------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# it takes a list of tuples as parameter
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('rf_model', RandomForestClassifier())
	])
pipeline.fit(X_train7, y_train7)
pp_pred3  = pipeline.predict(X_val3)
pp_proba3 = pipeline.predict_proba(X_val3)
SCORES(y_val3, pp_pred3, pp_proba3, "[Pipeline] ", cls_type=clstype)

