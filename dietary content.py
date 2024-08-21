import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Define a simple calorie estimation dictionary
calorie_dict = {
    'apple': 52,          # calories per 100g
    'banana': 89,         # calories per 100g
    'pizza': 266,         # calories per slice (average)
    'burger': 295,        # calories per burger (average)
}

def classify_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions

def get_calories(food_name):
    # Basic calorie estimation
    return calorie_dict.get(food_name.lower(), "Calorie information not available")

def main():
    img_path = 'path_to_your_image.jpg'  # Replace with your image path
    predictions = classify_image(img_path)
    
    print("Top predictions:")
    for pred in predictions:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")

    food_name = predictions[0][1]  # Take the top prediction
    calories = get_calories(food_name)
    print(f"Estimated calories for {food_name}: {calories}")

if __name__ == "__main__":
    main()
