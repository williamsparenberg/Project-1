import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, prescision_score, f1_score

#1 - Data Processing
df = pd.read_csv("Data/Project 1 Data.csv")

#2 - Data Visualization
X = df[['X','Y','Z']]
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

desc_stat = df[['X', 'Y', 'Z',]].describe()
desc_stat_step = df['Step'].value_counts().sort_index()
df.hist()

#3 - Data correlation
corr = df[['X','Y','Z']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#4 - Model Development with Pipelines
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models={}

#SVM Pipeline
svm_pipe = Pipeline([('scaler', StandardScaler()),('svm', SVC(random_state=42))])
svm_params = {'svm__C':[0.1, 1, 10, 100], 'svm__gamma':["scale", "auto"], 'svm__kernel':["linear", "poly", "rbf"]}

grid_svm = GridSearchCV(svm_pipe, svm_params, cv=5)
grid_svm.fit(X_train, y_train)
best_models["SVM"] = grid_svm.best_estimator_
print("\nSVM best params:", grid_svm.best_params_)

#Logistic Regression Pipeline
lr_pipe = Pipeline([('scaler', StandardScaler()),('lr', LogisticRegression(max_iter=1000, random_state=42))])
lr_params = {'lr__C':[0.1, 1, 10, 100], 'lr__solver':["lbfgs", "saga"]}

grid_lr = GridSearchCV(lr_pipe, lr_params, cv=5)
grid_lr.fit(X_train, y_train)
best_models["LogisticRegression"] = grid_lr.best_estimator_
print("\nLogistic Regression best params:", grid_lr.best_params_)

#Random Forest Pipeline with RandomSearchCV
rf_pipe = Pipeline([('scaler', StandardScaler()),('rf', RandomForestClassifier(random_state=42))])
rf_params = {'rf__n_estimators':[10, 50, 100, 200], 'rf__max_depth':[None, 5, 10, 15]}
randomsearch_rf = RandomizedSearchCV(rf_pipe, rf_params, cv=5, n_iter=15, random_state=42)


randomsearch_rf.fit(X_train, y_train)
best_models["RandomForest"] = randomsearch_rf.best_estimator_
print("\nLogistic Regression best params:", randomsearch_rf.best_params_)

#5 - Model Evaluation
for name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    
    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred, average="weighted")
    
    

