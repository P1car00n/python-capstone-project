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
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(repr(model).capitalize(), fontsize=16)

    axs[0].scatter(y_test, prediction)
    axs[0].set_xlabel('Actual Values')
    axs[0].set_ylabel('Predicted Values')
    axs[0].set_title('Scatter Plot: Actual vs. Predicted')

    residuals = [true - pred for true, pred in zip(y_test, prediction)]

    axs[1].hist(residuals, bins=10)
    axs[1].scatter(y_test, prediction)
    axs[1].set_xlabel('Residuals')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram: Residuals Distribution')

    plt.tight_layout()
    plt.show()
