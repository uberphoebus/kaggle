import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./diabetes.csv")
print(df.info())

# ---------------------------------------
# 1. str X   # 2. nan
# 우선 점수부터 보자
# ---------------------------------------
# model = DecisionTreeClassifier(random_state=1414)                 #0.6948051948051948
model = RandomForestClassifier(n_estimators=500,random_state=1414)  #0.7597402597402597
                                                                    #0.7792207792207793
정답지 = df['Outcome']
문제지 = df.drop('Outcome', axis=1)

문제지8, 문제지2 ,  정답지8, 정답지2 = train_test_split(문제지,정답지,
                 test_size=0.2,
                 random_state=1414,
                 shuffle=False)
model.fit(문제지8, 정답지8)
컴퓨터답2 = model.predict(문제지2)
score = accuracy_score(정답지2, 컴퓨터답2)
print("1차점수 : accuracy:", score)   #0.7662337662337663

df.cut

# ---------------------------------------
df.hist()
plt.show()


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
mm = MinMaxScaler()
정답지 = outcom
문제지
model.fit()
#---------------------------------
m.fit(문제지)         #비율계산
mm.transform(문제지)  #계산반영해
mm.fit_transform(문제지)

rs = RobustScaler()
ss = StandardScaler()
scalers = [MinMaxScaler(), RobustScaler(), StandardScaler()]


import warnings
warnings.filterwarnings(action='ignore')

import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, objective="binary:logistic")
# 규제(오버피팅방지)속성
# learning_rate=경사하강(Gradient Descent)

total_score = []

total_score / len(total_score)
np.array(total_score).mean()

train_test_split()  #stratify=y

from sklearn.model_selection import KFold, StratifiedKFold

kfold = KFold(n_splits=5) #random_state=1414, shuffle=False)
for train_index, test_index in kfold.split(문제지):
    학습용8, 검증용2 = 문제지.iloc[train_index], 문제지.iloc[test_index]
    학습용답안8, 검증용답안2 = 답안지.iloc[train_index], 답안지.iloc[test_index]

skfold = StratifiedKFold(n_splits=5) #random_state=1414, shuffle=False)
for train_index, test_index in skfold.split(문제지, 답안지):
    학습용8, 검증용2 = 문제지.iloc[train_index], 문제지.iloc[test_index]
    학습용답안8, 검증용답안2 = 답안지.iloc[train_index], 답안지.iloc[test_index]


from sklearn.model_selection import cross_val_score
#scores : ndarray  ::: 위와상동 == np.array(total_score)
total_score = cross_val_score(xgmodel, 문제지, 정답지, scoring='accuracy', cv=5)  #n_splits=cv=5
print(total_score)  #평균 0.8658772599949071










