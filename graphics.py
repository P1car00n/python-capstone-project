import pickle

import matplotlib.pyplot as plt
import numpy as np
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
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
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

    # Choose two features for visualization
    feature1 = "Efficiency"
    feature2 = "Total"
    feature1_index = X_train.columns.get_loc(feature1)
    feature2_index = X_train.columns.get_loc(feature2)

    # Extract the selected features from the training data
    X_train_selected = X_train.iloc[:, [feature1_index, feature2_index]].values

    # Fit the model with the selected features
    model.model.fit(X_train_selected, y_train)

    # Create a meshgrid of points to evaluate the decision boundary
    x_min, x_max = X_train_selected[:, 0].min(
    ) - 1, X_train_selected[:, 0].max() + 1
    y_min, y_max = X_train_selected[:, 1].min(
    ) - 1, X_train_selected[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(
            x_min, x_max, 0.1), np.arange(
            y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions for the grid points
    grid_predictions = model.get_prediction(grid_points)

    # Reshape the predictions to match the grid shape
    grid_predictions = grid_predictions.reshape(xx.shape)

    # Plot the decision boundary
    contour = axs[2].contourf(
        xx,
        yy,
        grid_predictions,
        cmap='rainbow',
        alpha=0.5)
    axs[2].scatter(X_train_selected[:, 0],
                   X_train_selected[:, 1], c=y_train, cmap='rainbow')
    axs[2].set_xlabel(feature1)
    axs[2].set_ylabel(feature2)
    axs[2].set_title('Decision Boundary')

    fig.colorbar(contour, axs[2])

    plt.tight_layout()
    plt.show()
