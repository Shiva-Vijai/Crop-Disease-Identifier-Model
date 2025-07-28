import tensorflow as tf

# Load your trained .h5 model
model = tf.keras.models.load_model("C:/Users/shiva/Desktop/test/rice_model_most_latest.h5")

# Convert to TFLite (default float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: model.tflite saved.")