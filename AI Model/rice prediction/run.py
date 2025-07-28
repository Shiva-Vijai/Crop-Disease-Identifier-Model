import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# ==== 1. Load the TFLite model ====
interpreter = tf.lite.Interpreter(model_path="rice_model.tflite")
interpreter.allocate_tensors()

# ==== 2. Get input and output details ====
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== 3. Set image size and class names ====
IMG_SIZE = (224, 224)
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 
               'Leaf Scald', 'Narrow Brown Leaf', 'Neck Blast', 'Rice Hispa', 
               'Sheath Blight', 'tungro']

# ==== 4. Load and preprocess the image ====
img_path = "image.png"  # Replace with your test image path
img = Image.open(img_path).resize(IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize to [0,1]
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# ==== 5. Run inference ====
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# ==== 6. Postprocess results ====
predicted_index = np.argmax(output_data)
predicted_label = class_names[predicted_index]
confidence = np.max(output_data) * 100

# ==== 7. Show prediction ====
plt.imshow(img)
plt.title(f"Prediction: {predicted_label}")
plt.axis("off")
plt.show()

print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")
