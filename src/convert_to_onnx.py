import tensorflow as tf
import tf2onnx
import os

# Load the trained model
model_path = os.path.join("..", "models", "cnn_lstm_crowd_model.h5")
model = tf.keras.models.load_model(model_path)

# Convert the model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Save the ONNX model
onnx_model_path = os.path.join("..", "models", "cnn_lstm_crowd_model.onnx")
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model successfully converted and saved to {onnx_model_path}")
