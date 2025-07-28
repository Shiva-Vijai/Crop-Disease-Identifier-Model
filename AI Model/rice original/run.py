
import numpy as np
import tensorflow as tf
# from keras.models import load_model
# from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ==== 1. Load the model ====
model = load_model("rice_model_most_latest.h5")  # Replace with your .h5 path

# ==== 2. Set image size and class names ====
IMG_SIZE = (224, 224)  # Must match what was used during training
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 
               'Leaf Scald', 'Narrow Brown Leaf', 'Neck Blast', 'Rice Hispa', 
               'Sheath Blight', 'tungro'] 

# ==== 3. Load and preprocess your test image ====
img_path = "image.png"  # Replace with your image path
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# ==== 4. Predict ====
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
predicted_label = class_names[predicted_index]
confidence = np.max(predictions) * 100

# ==== 5. Show result ====
plt.imshow(img)
plt.title(f"Prediction: {predicted_label}")
plt.axis("off")
plt.show()

print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")
