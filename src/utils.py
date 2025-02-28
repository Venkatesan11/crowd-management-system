import numpy as np

def normalize_data(data):
    """
    Normalize data to the range [0, 1].
    
    Args:
        data (numpy array): Input array to normalize.
    
    Returns:
        numpy array: Normalized data.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def reshape_for_lstm(X):
    """
    Reshape the data into a format suitable for LSTM input.
    
    Args:
        X (numpy array): Input data of shape (samples, features).
    
    Returns:
        numpy array: Reshaped data with an extra time step dimension.
    """
    return X.reshape((X.shape[0], 1, X.shape[1]))

# Example usage
if __name__ == "__main__":
    sample_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    normalized = normalize_data(sample_data)
    reshaped = reshape_for_lstm(normalized)

    print("Normalized Data:\n", normalized)
    print("Reshaped Data for LSTM:\n", reshaped)
