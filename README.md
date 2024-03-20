# stunning-carnival
sample
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Define a function to classify an image
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Load image and resize to match model input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    predictions = model.predict(img_array)  # Make predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Decode predictions
    return decoded_predictions

# Example usage
image_path = 'example_image.jpg'
predictions = classify_image(image_path)
print("Predictions for", image_path)
for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
