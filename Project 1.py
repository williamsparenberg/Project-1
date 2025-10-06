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
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

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
print("\nRandom Forest best params:", randomsearch_rf.best_params_)

#5 - Model Evaluation
for name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, F1 Score: {f1:.3f}")

#SVM Confusion Matrix
y_pred_svm = best_models["SVM"].predict(X_test)
cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("SVM Confusion Matrix")
plt.show()

#Logistic Regression Confusion Matrix
y_pred_lr = best_models["LogisticRegression"].predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

#Random Forest Confusion Matrix
y_pred_rf = best_models["RandomForest"].predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

#6 - Stacking Model
#RandomForest and Logistic Regression were chosen to stack
estimators= [('rf_pipe', svm_pipe), ('lr_pipe', lr_pipe)]
stacking_model= StackingClassifier(estimators=estimators, cv=5)
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)

acc_s = accuracy_score(y_test, y_pred_stacking)
prec_s = precision_score(y_test, y_pred_stacking, average="weighted")
f1_s = f1_score(y_test, y_pred_stacking, average="weighted")

print("\nStacking Model Performance:")
print(f"\nAccuracy: {acc_s:.3f}, Precision: {prec_s:.3f}, F1 Score: {f1_s:.3f}")
  
cm = confusion_matrix(y_test, y_pred_stacking)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Stacking Model Confusion Matrix")
plt.show()    

#7 - Model Evaluation
joblib.dump(stacking_model, "stacking_model.joblib")
model = joblib.load("stacking_model.joblib")
newdata = pd.DataFrame([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]], columns = ["X", "Y", "Z"])
predict_results = model.predict(newdata)

print("\nClassifications of New Data:")
print(predict_results)

