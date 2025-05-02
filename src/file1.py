import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


wine = load_wine()
x = wine.data 
y = wine.target 

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.10,random_state=42)


n_estimators = 10
params = {
    "max_depth":5,
    'n_estimators':10,
}

with mlflow.start_run():
    rf = RandomForestClassifier(**params,random_state=42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)


    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_params(params)
    # mlflow.set_tag("Training Info", "Basic RFC with wine data")

    print(accuracy)




