
# ------------------------------------------------
# iris
# ------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

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
print(iris_df['target'].unique())     # df[col].unique()
print(np.unique(iris['target'])) # np.unique(arr)

# 1. X, y sep
# 2. train_test sep
# 3. model selection
# 4. model.fit()
# 5. model.predict()
# 6. accuracy_score()

X = iris_df.drop('target', axis=1)
y = iris_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=10, shuffle=True, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))


# ------------------------------------------------
# toy_boston
# ------------------------------------------------

boston = load_boston()
print(boston.keys())
boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
print(boston_df.info())


# ------------------------------------------------
# bike sharing demand
# ------------------------------------------------

train = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/bike_sharing_demand/bike-sharing-demand/train.csv')
test  = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/bike_sharing_demand/bike-sharing-demand/test.csv')
sub   = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/bike_sharing_demand/bike-sharing-demand/sampleSubmission.csv')
train_test = train.append(test)
print(train_test.info()), print(train_test.shape)

# date
from datetime import datetime as dt
train_test['datetime'] = pd.to_datetime(train_test['datetime'])
print(train_test['datetime'][:5])

train_test['year'] = train_test['datetime'].dt.year
train_test['month'] = train_test['datetime'].dt.month
train_test['day'] = train_test['datetime'].dt.day
train_test['hour'] = train_test['datetime'].dt.hour
train_test['week'] = train_test['datetime'].dt.dayofweek
print(train_test[:5]), print(train_test.info())

# season check
print(train_test['season'].value_counts().sort_index())
print(train_test['season'].value_counts().sort_values())
# train_test['season'].value_counts().plot(kind='bar')
# plt.show()

# temp check
print(train_test['temp'].describe())
train_test['temp'].hist()
# plt.show()


# ------------------------------------------------
# binning ; temp, bins = [-100, 0, 13, 20, 27, 45, 100], labels = [1, 2, 3, 4, 5, 6]
# 1. df.loc[cond, band] = val
# 2. df[col].map(dict)
# 3. df[col].apply(lambda x: func(x))
# 4. pd.cut/qcut(df[cols], bins=[], labels=[])
# ------------------------------------------------

# temp_band1
train_test.loc[train_test['temp'] <= 0, 'temp_band1'] = 1
train_test.loc[(train_test['temp'] <= 13) & (train_test['temp'] >  0), 'temp_band1'] = 2
train_test.loc[(train_test['temp'] <= 20) & (train_test['temp'] > 13), 'temp_band1'] = 3
train_test.loc[(train_test['temp'] <= 27) & (train_test['temp'] > 20), 'temp_band1'] = 4
train_test.loc[(train_test['temp'] <= 45) & (train_test['temp'] > 27), 'temp_band1'] = 5
train_test.loc[train_test['temp'] > 45, 'temp_band1'] = 6
print(train_test[['temp', 'temp_band1']][:10])

# temp_band2
def temp_band(x):
    band = 0
    if x <= 0:
        band = 1
    elif (x <= 13) & (x >  0):
        band = 2
    elif (x <= 20) & (x > 13):
        band = 3
    elif (x <= 27) & (x > 20):
        band = 4
    elif (x <= 45) & (x > 27):
        band = 5
    else:
        band = 6
    return band
train_test['temp_band2'] = train_test['temp'].apply(lambda x: temp_band(x))
print(train_test[['temp', 'temp_band1', 'temp_band2']][:10])

# temp_band4
train_test['temp_band'] = pd.cut(train_test['temp'], bins=[-100, 0, 13, 20, 27, 45, 100], labels=[1, 2, 3, 4, 5, 6])
print(train_test[['temp', 'temp_band1', 'temp_band2', 'temp_band']][:10])
