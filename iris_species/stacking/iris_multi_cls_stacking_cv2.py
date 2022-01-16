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
from sklearn.model_selection import StratifiedKFold


def APPLY_STACKING(model, X_train7, y_train7, X_val3, folds=5) :
    train_fold_predict  = np.zeros((X_train7.shape[0], 1))
    test_predict  = np.zeros((X_val3.shape[0], folds))
    print(X_train7.shape, y_train7.shape)  #(120, 4) (120,)

    skFold = StratifiedKFold(n_splits=5, random_state=111, shuffle=True)
    for loop_cnt, (train_fold_idx, val_fold_idx) in enumerate(skFold.split(X_train7, y_train7)):
        X_train_fold5 = X_train7.iloc[train_fold_idx]
        y_train_fold5 = y_train7.iloc[train_fold_idx]
        X_val_fold5   = X_train7.iloc[val_fold_idx]
        y_val_fold5   = y_train7.iloc[val_fold_idx]
        #--------------------------------------------------------- fit    X_train_fold5, y_train_fold5
        model.fit(X_train_fold5, y_train_fold5)
        # --------------------------------------------------------- predict  X_val_fold5
        pred3_fold = model.predict(X_val_fold5)
        train_fold_predict[val_fold_idx , :]   = pred3_fold.reshape(-1, 1)
        # --------------------------------------------------------- predict   X_val3
        pred3_val  = model.predict(X_val3)
        test_predict[:, loop_cnt]  = pred3_val
    # --------------------------------------------------------- Average : predict X_val3
    test_predict_mean = np.mean(test_predict, axis=1)
    print(test_predict_mean.shape)
    test_predict_mean = test_predict_mean.reshape(-1, 1)
    print(test_predict_mean.shape)
    return  train_fold_predict, test_predict_mean


rf_model = RandomForestClassifier()
svc_model = SVC(probability=True)
lr_model = LogisticRegression()

rf_train_fold_predict, rf_test_predict_mean   = APPLY_STACKING(rf_model, X_train7, y_train7, X_val3)
svc_train_fold_predict, svc_test_predict_mean = APPLY_STACKING(svc_model, X_train7, y_train7, X_val3)
lr_train_fold_predict, lr_test_predict_mean   = APPLY_STACKING(lr_model, X_train7, y_train7, X_val3)

# --------------------------------------------------------- Average : predict X_val3
new_train_data = np.concatenate([rf_train_fold_predict, svc_train_fold_predict, lr_train_fold_predict], axis=1)
new_test_mean = np.concatenate([rf_test_predict_mean, svc_test_predict_mean, lr_test_predict_mean], axis=1)

print(new_train_data.shape)
print(new_train_data[:5])
print(new_test_mean.shape)
print(new_test_mean[:5])
#===============================================================================================================
xgb = XGBClassifier()
xgb.fit(new_train_data, y_train7)
xgb_pred = xgb.predict(new_test_mean)
xgb_proba = xgb.predict_proba(new_test_mean)

SCORES(y_val3, xgb_pred, xgb_proba, "[XGBClassifier] ", cls_type=clstype)
#===============================================================================================================
lgbm = LGBMClassifier()
lgbm.fit(new_train_data, y_train7)
lgbm_pred = lgbm.predict(new_test_mean)
lgbm_proba = lgbm.predict_proba(new_test_mean)

SCORES(y_val3, lgbm_pred, lgbm_proba, "[LGBMClassifier] ", cls_type=clstype)
