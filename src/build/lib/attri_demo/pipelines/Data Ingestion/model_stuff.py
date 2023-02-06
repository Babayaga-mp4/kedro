# Gets mlflow up and running.
# mlflow server - -backend - store - uri sqlite: ///db/mlflow.db - -default - artifact - root. / mlruns

import logging
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, precision_score, balanced_accuracy_score, recall_score,
                             log_loss, zero_one_loss)
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger()


def train_test_split_the_data(data):
    train ,test = train_test_split(data)

    train_x = train.drop(["Class"] ,axis = 1)
    test_x = test.drop(["Class"] ,axis = 1)

    train_y = train[["Class"]]
    test_y = test[["Class"]]
    return train_x, train_y, test_x, test_y

class_weights = {0:1, 1:15}

def train(train_x, train_y) -> LogisticRegression:
    lr = LogisticRegression(class_weight = class_weights, random_state = 42)
    lr.fit(train_x ,train_y)
    create_engine("sqlite:///mlflow.db", pool_pre_ping=True)
    mlflow.set_registry_uri(
        "sqlite:///mlflow.db")

    # print("Current tracking uri: {}".format(mlflow.get_tracking_uri()))
    # print("Current registry uri: {}".format(mlflow.get_registry_uri()))


    return lr


def eval_metrics(actual, raw_preds):
    pred=raw_preds.to_numpy()
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    accuracy=accuracy_score(actual, pred)
    report = precision_score(actual, pred)
    balanced_acc = balanced_accuracy_score(actual, pred)
    recall = recall_score(actual, pred)
    lloss = log_loss(actual, pred)
    zoloss = zero_one_loss(actual, pred)


    df={}
    df["rmse"]=rmse
    df["mae"]=mae
    df["r2"]=r2
    df["accuracy_value"]=accuracy
    df['precision_score'] = report
    df['balanced_acc'] = balanced_acc
    df['recall'] = recall
    df['lloss'] = lloss
    df['zoloss'] = zoloss

    return pd.DataFrame(df,index=[0])

def prediction(lr, test_x):
    predicted_qualities = lr.predict(test_x)
    return pd.DataFrame(predicted_qualities ,columns=['Predictions'])

def mlflow_logging(model,data):
            mlflow.log_param('class_weights', class_weights)
            mlflow.log_metric("accuracy_score" ,data.iloc[0]['accuracy_value'])
            mlflow.log_metric("precision" ,data.iloc[0]['precision_score'])
            mlflow.log_metric("rmse" ,data.iloc[0]['rmse'])
            mlflow.log_metric("log_loss" ,data.iloc[0]['lloss'])
            mlflow.log_metric("r2" ,data.iloc[0]['r2'])
            mlflow.log_metric("zero-one_loss" ,data.iloc[0]['zoloss'])
            mlflow.log_metric("mae" ,data.iloc[0]['mae'])
            mlflow.log_metric("balanced_accuracy_score" ,data.iloc[0]['balanced_acc'])
            mlflow.log_metric("recall_score" ,data.iloc[0]['recall'])
            mlflow.sklearn.log_model(
                sk_model = model,
                artifact_path = "sklearn-model")

            model_uri = "runs:/{}/sklearn-model".format(mlflow.active_run().info.run_id)
            mlflow.register_model(model_uri ,"Fraud Detection-2")

          
