import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# getting the data
dataset_train = pd.read_parquet(
    input('Path to the train dataset: '))
dataset_test = pd.read_parquet(
    input('Path to the test dataset: '))
X_train = dataset_train.drop(columns='Place')
y_train = dataset_train['Place']
X_test = dataset_test.drop(columns='Place')
y_test = dataset_test['Place']

# Create an instance of RandomForestRegressor
model = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

# Fit the model on our training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create a new RandomForestRegressor model with the best hyperparameters
best_model = RandomForestRegressor(**best_params)

# Fit the best model on our training data
best_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_model.predict(X_test)

print(best_params)
