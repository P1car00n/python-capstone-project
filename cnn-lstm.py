import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential

# getting the data
dataset_train = pd.read_parquet(
    input('Path to the train dataset: '))
dataset_test = pd.read_parquet(
    input('Path to the test dataset: '))
X_train = dataset_train.drop(columns='Place')
y_train = dataset_train['Place']
X_test = dataset_test.drop(columns='Place')
y_test = dataset_test['Place']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15)

sequence_length = X_train.shape[0]
num_features = X_train.shape[1]

# Reshape the input data to match the expected shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))

# Define our CNN-LSTM model architecture
model = Sequential()
model.add(
    Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=(
            X_train.shape[1],
            1)))
# model.add(Flatten())
# model.add(Reshape((-1, 368640)))  # Reshape the flattened output to
# (batch_size, timesteps, features)
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# Train the model
model_hist = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(
        X_val,
        y_val))

# Evaluate the model
model_eval = model.evaluate(X_test, y_test)
print("Model Accuracy:", str(round(model_eval * 100, 2)) + '%')

plt.plot(model_hist.history['loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='lower left')

plt.show()

# Make predictions
predictions = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
