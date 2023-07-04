import pickle

import numpy as np
import pandas as pd
from sklearn import metrics

from models import GBR, LRM, RFR, RidgeModel

# getting the data
dataset_train = pd.read_parquet(
    input('Path to the train dataset: '))
dataset_test = pd.read_parquet(
    input('Path to the test dataset: '))
X_train = dataset_train.drop(columns='Place')
y_train = dataset_train['Place']
X_test = dataset_test.drop(columns='Place')
y_test = dataset_test['Place']

models = []
with open(input('Path to the models\' file : '), "rb") as file:
    models = pickle.load(file)

# Make predictions on the test set for each model
predictions = [model.get_prediction(X_test) for model in models]

for (model, prediction) in zip(models, predictions):
    train_prediction = model.get_prediction(X_train)

    # MAE
    mae_train = metrics.mean_absolute_error(y_train, train_prediction)
    mae_test = metrics.mean_absolute_error(y_test, prediction)
    print(
        'Mean Absolute Error for',
        model,
        'for train values is',
        mae_train,
        'and for test values is',
        mae_test)

    # MSE
    mse_train = metrics.mean_squared_error(y_train, train_prediction)
    mse_test = metrics.mean_squared_error(y_test, prediction)
    print(
        'Mean Squared Error for',
        model,
        'for train values is',
        mse_train,
        'and for test values is',
        mse_test)
    
    # RMSE
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    print(
        'Root Mean Squared Error for',
        model,
        'for train values is',
        rmse_train,
        'and for test values is',
        rmse_train)

    # R^2
    r2_train = metrics.r2_score(y_train, train_prediction)
    r2_test = metrics.r2_score(y_test, prediction)
    print(
        'R-squared for',
        model,
        'for train values is',
        rmse_train,
        'and for test values is',
        rmse_train)
