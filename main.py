# This Python 3 environment comes with many helpful analytics libraries installed
from typing import List

import matplotlib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # to plot charts
import seaborn as sns  # used for data visualization
import warnings  # avoid warning flash

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset
# Input data files are available in the read-only "../input/" directory
import os

for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("input/diabetes.csv")

# Exploratory Data Analysis
# Analyzing the dataset
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.describe())

df1 = df.copy()
# Cleaning the dataset
# dropping duplicate values
df1 = df1.drop_duplicates()
# check for missing values
df1.isnull().sum()
# checking for 0 values in 5 columns , Age & DiabetesPedigreeFunction do not have minimum 0 value so no need to replace , also no. of pregnancies as 0 is possible as observed in df.describe
print(df1[df1['BloodPressure'] == 0].shape[0])
print(df1[df1['Glucose'] == 0].shape[0])
print(df1[df1['SkinThickness'] == 0].shape[0])
print(df1[df1['Insulin'] == 0].shape[0])
print(df1[df1['BMI'] == 0].shape[0])
# replacing 0 values with median of that column
df1['Glucose'] = df1['Glucose'].replace(0, df1['Glucose'].mean())
df1['BloodPressure'] = df1['BloodPressure'].replace(0, df1['BloodPressure'].mean())
df1['SkinThickness'] = df1['SkinThickness'].replace(0, df1['SkinThickness'].mean())
df1['Insulin'] = df1['Insulin'].replace(0, df1['Insulin'].mean())
df1['BMI'] = df1['BMI'].replace(0, df1['BMI'].mean())
df1['Age'] = df1['Age'].replace(0, df1['Age'].mean())

# Data Visualization
labels = ['Healthy', 'Diabetic']
sizes = [*df.Outcome.value_counts()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax = plt.axis('equal')
# histogram for each feature
df.hist(bins=10, figsize=(10, 10))
plt.show()
plt.figure(figsize=(16, 12))
sns.set_style(style='whitegrid')
plt.subplot(3, 3, 1)
sns.boxplot(x='Glucose', data=df1)
plt.subplot(3, 3, 2)
sns.boxplot(x='BloodPressure', data=df1)
plt.subplot(3, 3, 3)
sns.boxplot(x='Insulin', data=df1)
plt.subplot(3, 3, 4)
sns.boxplot(x='BMI', data=df1)
plt.subplot(3, 3, 5)
sns.boxplot(x='Age', data=df1)
plt.subplot(3, 3, 6)
sns.boxplot(x='SkinThickness', data=df1)
plt.subplot(3, 3, 7)
sns.boxplot(x='Pregnancies', data=df1)
plt.subplot(3, 3, 8)
sns.boxplot(x='DiabetesPedigreeFunction', data=df1)
scatter_matrix(df1, figsize=(30, 30))
plt.show()

# Feature Selection
corrmat = df1.corr()
sns.heatmap(corrmat, annot=True)
df1 = df1.drop(['BloodPressure', 'DiabetesPedigreeFunction'], axis='columns')
plt.show()

# Handling Outliers
x = df1
quantile = QuantileTransformer(n_quantiles=768)
X = quantile.fit_transform(x)
df3 = pd.DataFrame(X)
df3.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Insulin', 'Age', 'Outcome']
print(df1.head())

df2 = df.copy()
# dropping rows have missing or 0 values
df2['Glucose'] = df2['Glucose'].replace(0, np.nan)
df2['BloodPressure'] = df2['BloodPressure'].replace(0, np.nan)
df2['SkinThickness'] = df2['SkinThickness'].replace(0, np.nan)
df2['Insulin'] = df2['Insulin'].replace(0, np.nan)
df2['BMI'] = df2['BMI'].replace(0, np.nan)
df2 = df2.dropna()

print(df2.info())

# Feature Selection
corrmat = df2.corr()
sns.heatmap(corrmat, annot=True)
df2 = df2.drop(['BloodPressure', 'DiabetesPedigreeFunction', 'SkinThickness'], axis='columns')
plt.show()

# Handling Outliers
x = df2
quantile = QuantileTransformer(n_quantiles=392)
X = quantile.fit_transform(x)
df2 = pd.DataFrame(X)
df2.columns = ['Pregnancies', 'Glucose', 'Insulin', 'BMI',
               'Age', 'Outcome']
print(df2.head())

plt.figure(figsize=(16, 12))
sns.set_style(style='whitegrid')
plt.subplot(3, 3, 3)
sns.boxplot(x=df2['Pregnancies'], data=df2)
plt.subplot(3, 3, 1)
sns.boxplot(x=df2['Glucose'], data=df2)
plt.subplot(3, 3, 2)
sns.boxplot(x=df2['BMI'], data=df2)
plt.subplot(3, 3, 4)
sns.boxplot(x=df2['Age'], data=df2)

plt.show()
dataList = [df, df1, df2]
acc_mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
f1s_mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
pre_mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
rec_mat = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

i = 0
while i < 3:
    data = dataList[i]

    # Split the data frame
    target_name = 'Outcome'
    y = data[target_name]  # given predictions - training data
    X = data.drop(target_name, axis=1)  # dropping the Outcome column and keeping all other columns as X
    print(X.head())
    print(y.head())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112)

    # Classification algorithms

    # KNN
    print('--------------------------------------------------')
    # List Hyperparameters to tune
    knn = KNeighborsClassifier()
    knn_n_neighbors = list(range(1, 30))
    knn_leaf_size = [30]
    knn_algorithms = ['auto', 'kd_tree', 'ball_tree', 'brute']
    knn_metric = ['euclidean', 'manhattan', 'minkowski']
    # convert to dictionary
    hyperP_knn = dict(n_neighbors=knn_n_neighbors, leaf_size=knn_leaf_size, algorithm=knn_algorithms, metric=knn_metric)
    # Making model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=12)
    grid_search = GridSearchCV(estimator=knn, param_grid=hyperP_knn, n_jobs=-1, cv=cv, scoring='f1',
                               error_score=0)
    best_model = grid_search.fit(X_train, y_train)
    # Best Hyperparameters Value
    print('Best algorithm:', best_model.best_estimator_.get_params()['algorithm'])
    print('Best metric:', best_model.best_estimator_.get_params()['metric'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    # Predict testing set
    knn_pred = best_model.predict(X_test)
    print("KNearestNeighborhood Classification Report is:\n", classification_report(y_test, knn_pred))
    print("\n F1:\n", f1_score(y_test, knn_pred))
    print("\n Precision score is:\n", precision_score(y_test, knn_pred))
    print("\n Recall score is:\n", recall_score(y_test, knn_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
    acc_mat[i][0] = accuracy_score(y_test, knn_pred)
    f1s_mat[i][0] = f1_score(y_test, knn_pred)
    pre_mat[i][0] = precision_score(y_test, knn_pred)
    rec_mat[i][0] = recall_score(y_test, knn_pred)

    # SVM
    print('--------------------------------------------------')
    # List Hyperparameters to tune
    svm = SVC()
    svm_kernel = ['rbf', 'sigmoid', 'poly']
    svm_C = [50, 10, 1.0, 0.1, 0.01]
    svm_gamma = ['scale']
    # convert to dictionary
    hyperP_svm = dict(kernel=svm_kernel, C=svm_C, gamma=svm_gamma)
    # define grid search
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=12)
    grid_search = GridSearchCV(estimator=svm, param_grid=hyperP_svm, n_jobs=-1, cv=cv, scoring='f1',
                               error_score=0)
    svm_best_model = grid_search.fit(X_train, y_train)
    svm_pred = svm_best_model.predict(X_test)
    print('Best kernel:', svm_best_model.best_estimator_.get_params()['kernel'])
    print('Best C:', svm_best_model.best_estimator_.get_params()['C'])
    print("Support Vector Machine Classification Report is:\n", classification_report(y_test, svm_pred))
    print("\n F1:\n", f1_score(y_test, svm_pred))
    print("\n Precision score is:\n", precision_score(y_test, svm_pred))
    print("\n Recall score is:\n", recall_score(y_test, svm_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
    acc_mat[i][1] = accuracy_score(y_test, svm_pred)
    f1s_mat[i][1] = f1_score(y_test, svm_pred)
    pre_mat[i][1] = precision_score(y_test, svm_pred)
    rec_mat[i][1] = recall_score(y_test, svm_pred)

    # DT
    print('--------------------------------------------------')
    dt = DecisionTreeClassifier()
    # Create the parameter grid based on the results of random search
    dt_max_depth = [5, 10, 20, 30]
    dt_min_samples_leaf = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    dt_criterion = ["gini", "entropy"]
    dt_random_state = [66]
    # convert to dictionary
    hyperP_dt = dict(max_depth=dt_max_depth, min_samples_leaf=dt_min_samples_leaf,
                     criterion=dt_criterion, random_state=dt_random_state)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=12)
    grid_search = GridSearchCV(estimator=dt,
                               param_grid=hyperP_dt,
                               cv=cv, n_jobs=-1, verbose=1, scoring="f1")
    best_model = grid_search.fit(X_train, y_train)
    dt_pred = best_model.predict(X_test)
    print('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
    print('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
    print('Best criterion:', best_model.best_estimator_.get_params()['criterion'])
    print("Decision Tree Classification Report is:\n", classification_report(y_test, dt_pred))
    print("\n F1:\n", f1_score(y_test, dt_pred))
    print("\n Precision score is:\n", precision_score(y_test, dt_pred))
    print("\n Recall score is:\n", recall_score(y_test, dt_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
    acc_mat[i][2] = accuracy_score(y_test, dt_pred)
    f1s_mat[i][2] = f1_score(y_test, dt_pred)
    pre_mat[i][2] = precision_score(y_test, dt_pred)
    rec_mat[i][2] = recall_score(y_test, dt_pred)

    # RF
    print('--------------------------------------------------')
    # define models and parameters
    rf = RandomForestClassifier()
    rf_n_estimators = [250]
    rf_max_features = ['sqrt']
    rf_min_samples_leaf = [5, 15, 25, 35, 45, 50]
    rf_criterion = ['entropy', 'gini', 'log_loss']
    # convert to dictionary
    hyperP_rf = dict(n_estimators=rf_n_estimators, max_features=rf_max_features,
                     min_samples_leaf=rf_min_samples_leaf, criterion=rf_criterion)
    # define grid search
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=12)
    grid_search = GridSearchCV(estimator=rf, param_grid=hyperP_rf, n_jobs=-1, cv=cv, scoring='f1', error_score=0)
    best_model = grid_search.fit(X_train, y_train)
    rf_pred = best_model.predict(X_test)
    print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
    print('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
    print("Random Forest Classification Report is:\n", classification_report(y_test, rf_pred))
    print("\n F1:\n", f1_score(y_test, rf_pred))
    print("\n Precision score is:\n", precision_score(y_test, rf_pred))
    print("\n Recall score is:\n", recall_score(y_test, rf_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
    acc_mat[i][3] = accuracy_score(y_test, rf_pred)
    f1s_mat[i][3] = f1_score(y_test, rf_pred)
    pre_mat[i][3] = precision_score(y_test, rf_pred)
    rec_mat[i][3] = recall_score(y_test, rf_pred)

    i = i + 1

# Accuracy comparison
xac_axis = ['KNN', 'SVM', 'DT', 'RF', ]
yac1_axis = [acc_mat[0][0], acc_mat[0][1], acc_mat[0][2], acc_mat[0][3]]
yac2_axis = [acc_mat[1][0], acc_mat[1][1], acc_mat[1][2], acc_mat[1][3]]
yac3_axis = [acc_mat[2][0], acc_mat[2][1], acc_mat[2][2], acc_mat[2][3]]
plt.plot(xac_axis, yac1_axis, "-r", label="D1")
plt.plot(xac_axis, yac2_axis, "-b", label="D2")
plt.plot(xac_axis, yac3_axis, "-g", label="D3")
plt.title('Accuracy Comparison')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.show()

# Precision comparison
xpr_axis = ['KNN', 'SVM', 'DT', 'RF', ]
ypr1_axis = [pre_mat[0][0], pre_mat[0][1], pre_mat[0][2], pre_mat[0][3]]
ypr2_axis = [pre_mat[1][0], pre_mat[1][1], pre_mat[1][2], pre_mat[1][3]]
ypr3_axis = [pre_mat[2][0], pre_mat[2][1], pre_mat[2][2], pre_mat[2][3]]
plt.plot(xpr_axis, ypr1_axis, "-r", label="D1")
plt.plot(xpr_axis, ypr2_axis, "-b", label="D2")
plt.plot(xpr_axis, ypr3_axis, "-g", label="D3")
plt.title('Precision Comparison')
plt.xlabel('Classifier')
plt.ylabel('Precision')
plt.legend(loc="upper left")
plt.show()

# Rreecall comparison
xre_axis = ['KNN', 'SVM', 'DT', 'RF', ]
yre1_axis = [rec_mat[0][0], rec_mat[0][1], rec_mat[0][2], rec_mat[0][3]]
yre2_axis = [rec_mat[1][0], rec_mat[1][1], rec_mat[1][2], rec_mat[1][3]]
yre3_axis = [rec_mat[2][0], rec_mat[2][1], rec_mat[2][2], rec_mat[2][3]]
plt.plot(xre_axis, yre1_axis, "-r", label="D1")
plt.plot(xre_axis, yre2_axis, "-b", label="D2")
plt.plot(xre_axis, yre3_axis, "-g", label="D3")
plt.title('Recall Comparison')
plt.xlabel('Classifier')
plt.ylabel('Recall')
plt.legend(loc="upper left")
plt.show()

# F1-score comparison
xf_axis = ['KNN', 'SVM', 'DT', 'RF', ]
yf1_axis = [f1s_mat[0][0], f1s_mat[0][1], f1s_mat[0][2], f1s_mat[0][3]]
yf2_axis = [f1s_mat[1][0], f1s_mat[1][1], f1s_mat[1][2], f1s_mat[1][3]]
yf3_axis = [f1s_mat[2][0], f1s_mat[2][1], f1s_mat[2][2], f1s_mat[2][3]]
plt.plot(xf_axis, yf1_axis, "-r", label="D1")
plt.plot(xf_axis, yf2_axis, "-b", label="D2")
plt.plot(xf_axis, yf3_axis, "-g", label="D3")
plt.title('F1score Comparison')
plt.xlabel('Classifier')
plt.ylabel('F1score')
plt.legend(loc="upper left")
plt.show()
