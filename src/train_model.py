import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, TimeDistributed
import numpy as np
import os

# Load processed data
X_train = np.load(os.path.join("..", "data", "processed_data", "X_train.npy"))
y_train = np.load(os.path.join("..", "data", "processed_data", "y_train.npy"))

# Define the CNN + LSTM model
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(10, 64, 64, 1)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Save the trained model
model.save(os.path.join("..", "models", "cnn_lstm_crowd_model.h5"))

print("Model training complete and saved as cnn_lstm_crowd_model.h5")
