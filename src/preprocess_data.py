import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Load dataset (PETS2009 annotations)
data_path = os.path.join("..", "data", "dataset", "annotations.csv")
df = pd.read_csv(data_path)

# Extract bounding box features
X = df[['x_min', 'y_min', 'x_max', 'y_max']].values
y = df['crowd_count'].values

# Normalize the input data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Reshape X for CNN-LSTM input (assuming 10 time steps, 64x64 images)
X_reshaped = X_normalized.reshape(-1, 10, 64, 64, 1)  # Adjust as per dataset needs

# Save processed data
processed_data_path = os.path.join("..", "data", "processed_data")
os.makedirs(processed_data_path, exist_ok=True)

np.save(os.path.join(processed_data_path, "X_train.npy"), X_reshaped)
np.save(os.path.join(processed_data_path, "y_train.npy"), y)

print("Data preprocessing complete. Processed data saved.")
