
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import warnings
sns.set()
warnings.filterwarnings('ignore')

# https://www.kaggle.com/shrutimechlearn/step-by-step-diabetes-classification-knn-detailed
# https://www.kaggle.com/vincentlugat/pima-indians-diabetes-eda-prediction-0-906

# OSEMN Pipeline
# Obtaining
# Scrubbing / Cleaning
# Exploring / Visualizing
# Modeling
# INterpreting

diabetes_data = pd.read_csv('./input/diabetes.csv')

print(diabetes_data.head())
print(diabetes_data.info(verbose=True))
print(diabetes_data.describe())

# COLS
# Pregnancies   임신 횟수
# Glucose       혈당
# BloodPressure 혈압
# SkinThickness 삼두 두께
# Insulin       인슐린
# BMI           weight in kg/(height in m)^2
# DiabetesPedigreeFunction 당뇨병 가족력 상관관계 지수
# Age
# Outcome       당뇨 = 1

# 0 data check
print(diabetes_data[diabetes_data == 0].count())
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
cols_drop = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age']
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# 0 to nan
diabetes_copy = diabetes_data.copy(deep=True)
diabetes_copy[zero_cols] = diabetes_copy[zero_cols].replace(0, np.nan)
print(diabetes_copy[zero_cols].isna().sum())
# diabetes_data.hist(figsize=(7, 7))
# plt.show()
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace=True)
diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace=True)
diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].median(), inplace=True)
diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].median(), inplace=True)
diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].median(), inplace=True)

# balance check
print(diabetes_copy.shape)
print(diabetes_copy.Outcome.value_counts())
# diabetes_copy.Outcome.value_counts().plot(kind='bar')
# plt.show()

# scatter matrix
from pandas.plotting import scatter_matrix
# scatter_matrix(diabetes_data, figsize=(10, 10))

# pairplot
# sns.pairplot(diabetes_data, hue='Outcome')

# heatmap ; 상관관계 확인
plt.figure(figsize=(8, 6))
# sns.heatmap(diabetes_data.corr(), annot=True)
# plt.show()

# scaling ; 같은 기준으로 비교
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_fit = sc_X.fit_transform(diabetes_data.drop(['Outcome'], axis=1))
X = pd.DataFrame(X_fit, columns=cols_drop)
print(X.head())

# y
y = diabetes_data.Outcome
print(y.head())

# train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,
                                                    random_state=42,
                                                    stratify=y)
# DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# accuracy_score
from sklearn.metrics import accuracy_score
dt_ascore = accuracy_score(y_test, dt_pred)
print(dt_ascore)

# RandomForestClassifier

# KNN(Kn