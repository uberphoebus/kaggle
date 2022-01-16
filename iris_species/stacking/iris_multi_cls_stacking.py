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


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

rf_model = RandomForestClassifier()
svc_model = SVC(probability=True)
lr_model = LogisticRegression()

rf_model.fit(X_train7, y_train7)
svc_model.fit(X_train7, y_train7)
lr_model.fit(X_train7, y_train7)

rf_pred3 = rf_model.predict(X_val3)
svc_pred3 = svc_model.predict(X_val3)
lr_pred3 = lr_model.predict(X_val3)
print(rf_pred3.shape,  svc_pred3.shape, lr_pred3.shape)

rf_proba3 = rf_model.predict_proba(X_val3)
svc_proba3 = svc_model.predict_proba(X_val3)
lr_proba3 = lr_model.predict_proba(X_val3)

SCORES(y_val3, rf_pred3, rf_proba3, "[RandomForestClassifier] ", cls_type=clstype)
SCORES(y_val3, svc_pred3, svc_proba3, "[SVC] ", cls_type=clstype)
SCORES(y_val3, lr_pred3, lr_proba3, "[LogisticRegression] ", cls_type=clstype)

#===============================================================================================================
new_train_data333 = np.array([rf_pred3, svc_pred3, lr_pred3])
print(new_train_data333.shape)


new_train_data333 = new_train_data333.T
print(new_train_data333.shape)
print(new_train_data333[:5])
#===============================================================================================================
xgb = XGBClassifier()
xgb.fit(new_train_data333, y_val3)
xgb_pred = xgb.predict(new_train_data333)
xgb_proba = xgb.predict_proba(new_train_data333)

SCORES(y_val3, xgb_pred, xgb_proba, "[XGBClassifier] ", cls_type=clstype)
#===============================================================================================================
lgbm = LGBMClassifier()
lgbm.fit(new_train_data333, y_val3)
lgbm_pred = lgbm.predict(new_train_data333)
lgbm_proba = lgbm.predict_proba(new_train_data333)

SCORES(y_val3, lgbm_pred, lgbm_proba, "[LGBMClassifier] ", cls_type=clstype)


from sklearn.ensemble import VotingClassifier
vot_model  = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm)],
                              voting='soft',)
vot_model.fit(new_train_data333, y_val3)
pred = vot_model.predict(new_train_data333)
proba = vot_model.predict_proba(new_train_data333)
SCORES(y_val3, pred, proba, "[Stacking Ensemble votting] ",  cls_type=clstype)