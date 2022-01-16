
# ------------------------------------------------
# 코딩테스트 연습
# ------------------------------------------------
# skl search dataset

import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris

warnings.filterwarnings(action='ignore')

dataset = load_iris()   # array type

# print(dataset['data'])          # array type
# print(dataset['target'])        # array type
# print(dataset.keys())           # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
# print(dataset['target_names'])  # ['setosa' 'versicolor' 'virginica']
# print(dataset['feature_names']) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# iris  ; 꽃잎, 꽃받침의 넓이, 길이에 따라 종 구분, binary가 아닌 multi
# to df ; 꽃잎 넓이/길이, 꽃받침 넓이/길이, 타겟, 150 rows

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.DataFrame(data=dataset['data'], columns=features)
df_target = pd.DataFrame(data=dataset['target'], columns=['target'])
# df = pd.concat([df, df_target], axis=1)
df['target'] = pd.DataFrame(data=dataset['target'], columns=['target'])

# print(df.head())
# print(df.shape)   # (150, 5)
print(df.info())

# ------------------------------------------------
print(dataset['target'][:10])
print(dataset['target'][-10:])

# ------------------------------------------------
print(df['target'].unique())                           # df.unique()
print(np.unique(dataset['target'],return_counts=True)) # np.unique(arr)

# ------------------------------------------------
# pd.DataFrame(data=, index=, columns=)
# df.columns = [1, 2, 3, 4, 5]
# df.rename(index=, columns={'before':'atfer'}, in)
# df.rename({'before':'after'}, axis=)
# print(df.columns)
# print(df.index.name)
# print(df.index.values)

# ------------------------------------------------
# df + arr
df['target'] = dataset['target'] # df[col] = arr

# df + srs ; pd.concat([dfs], axis=1)
s = pd.Series(dataset['target'])
df = pd.concat([df, s], axis=1)
df.rename(columns={0:'target1'}, inplace=True)
df.drop('target1', axis=1, inplace=True)
# df.drop(1, inplace=True)



# ------------------------------------------------
# 1. X, y sep
# 2. train_test sep
# 3. model selection
# 4. model.fit(), model.predict(), accuracy_score()

# ------------------------------------------------
# uci credit card

# uci = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/creditcard/UCI_Credit_Card.csv')
# X = uci.drop('default.payment.next.month', axis=1)
# y = uci['default.payment.next.month']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
#
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# score = accuracy_score(y_test, y_pred)


# ------------------------------------------------
dataset = load_boston()

# df = pd.DataFrame(data=dataset['data'], columns=dataset['feature_names'])
# df['target'] = dataset['target']
#
# # X, y sep
# X = df.drop(['target'], axis=1)
# y = df['target']
#
# # train_test sep
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5458, shuffle=True)



# ------------------------------------------------
# bike

train = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/bike_sharing_demand/bike-sharing-demand/train.csv')
test  = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/bike_sharing_demand/bike-sharing-demand/test.csv')
sub   = pd.read_csv('C:/AI/pythonProject/venv/0_kaggle/bike_sharing_demand/bike-sharing-demand/sampleSubmission.csv')
train_test = train.append(test)

# date ; slicing or regex
train_test['datetime'] = pd.to_datetime(train_test['datetime'])
# train_test['datetime'] = train_test['datetime'].astype('datetime64')
train_test['year'] = train_test['datetime'].dt.year
train_test['month'] = train_test['datetime'].dt.month
train_test['day'] = train_test['datetime'].dt.day
train_test['hour'] = train_test['datetime'].dt.hour
train_test['week'] = train_test['datetime'].dt.dayofweek

# season count
print(train_test.groupby('season').count())
print(train_test['season'].value_counts().sort_index())
print(train_test['season'].value_counts().sort_values())

# train_test['season'].value_counts().plot(kind='bar')
# plt.show()

pd.set_option('display.max_columns', 100)

print(train_test['temp'].describe())
train_test.temp.hist()
# plt.show()

# binning
# 1. df.loc[cond, band] = val
# 2. dict = {}, df[col].map(dict)
# 3. lambda
# 4. cut/qcut

def temp_bind(x):
    temp_band = 0
    if x <= 0:
        temp_band = 1
    elif (x > 0) & (x <= 13):
        temp_band = 2
    elif (x > 13) & (x <= 20):
        temp_band = 3
    elif (x > 20) & (x <= 27):
        temp_band = 4
    elif (x > 27) & (x <= 45):
        temp_band = 5
    else:
        temp_band = 6
    return temp_band
train_test['temp_band2'] = train_test.temp.apply(lambda x: temp_bind(x))

train_test['temp_band'] = pd.cut(train_test.temp,
                                 bins=[-100, 0, 13, 20, 27, 45, 100],
                                 labels=[1, 2, 3, 4, 5, 6])

print(train_test.temp_band.unique())
print(train_test.temp_band2.unique())
print(train_test[['temp', 'temp_band', 'temp_band2']][:9])


print('-' * 50)
# print(train_test.info())
# print(train_test[:5])
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------