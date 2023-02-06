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




          
