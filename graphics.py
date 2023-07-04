import pickle

import matplotlib.pyplot as plt
import pandas as pd

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
    plt.scatter(y_test, prediction)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot: Actual vs. Predicted')
    plt.show()