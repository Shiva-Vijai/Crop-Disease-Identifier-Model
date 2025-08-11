import os
from flask import Flask, render_template, request, redirect, url_for
import webview
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__,static_folder='./static',template_folder="./templates")

STATIC_FOLDER_user_files = 'static/user_files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

disease_solutions = {
    "rice": {
        'Bacterial Leaf Blight': 'Use resistant varieties, ensure proper field drainage, and apply recommended bactericides.',
        'Brown Spot': 'Apply balanced fertilizers, avoid water stress, and use fungicides if severe.',
        'Healthy': 'No issues detected — maintain good irrigation and nutrient management.',
        'Leaf Blast': 'Plant resistant varieties, avoid excessive nitrogen, and use fungicides as needed.',
        'Leaf Scald': 'Improve water management and remove infected debris.',
        'Narrow Brown Leaf': 'Apply balanced nutrients and improve drainage; consider fungicides if severe.',
        'Neck Blast': 'Plant resistant varieties, manage nitrogen, and apply fungicide at booting stage.',
        'Rice Hispa': 'Handpick beetles early, avoid over-fertilization, and use insecticides if infestation is high.',
        'Sheath Blight': 'Reduce plant density, manage nitrogen, and use fungicides at early symptoms.',
        'tungro': 'Control green leafhopper vector with insecticides and use resistant varieties.'
    },

    "wheat": {
        'Aphid': 'Spray insecticides and encourage natural predators like lady beetles.',
        'Black Rust': 'Plant resistant varieties and spray fungicides at early detection.',
        'Blast': 'Avoid late planting, ensure proper drainage, and apply fungicides.',
        'Brown Rust': 'Use resistant varieties and apply fungicides at first sign.',
        'Common Root Rot': 'Practice crop rotation and use seed treatment fungicides.',
        'Fusarium Head Blight': 'Use resistant varieties, avoid overhead irrigation during flowering, and apply fungicide.',
        'Healthy': 'No disease detected — maintain regular monitoring and good field hygiene.',
        'Leaf Blight': 'Use resistant seeds and apply fungicides if needed.',
        'Mildew': 'Ensure good air circulation and apply fungicides.',
        'Mite': 'Use miticides and avoid water stress.',
        'Septoria': 'Rotate crops, remove debris, and use fungicides.',
        'Smut': 'Use certified treated seeds and resistant varieties.',
        'Stem Fly': 'Use insecticides and avoid staggered sowing.',
        'Tan Spot': 'Remove crop debris and apply fungicides early.',
        'Yellow Rust': 'Plant resistant varieties and spray fungicides promptly.'
    },

    "tomato": {
        'bacterial spot': 'Remove infected leaves, avoid overhead watering, and apply copper-based sprays.',
        'early blight': 'Rotate crops, mulch plants, and use fungicides.',
        'healthy': 'No disease found — maintain proper care and regular inspection.',
        'late blight': 'Remove affected plants and apply fungicides immediately.',
        'leaf mold': 'Improve ventilation, reduce humidity, and apply fungicides.',
        'septoria leaf spot': 'Remove infected leaves and apply fungicides.',
        'spider mites': 'Spray water on undersides of leaves and use miticides.',
        'target spot': 'Remove infected foliage and apply fungicides.',
        'mosaic virus': 'Remove infected plants and control insect vectors.',
        'leaf curl virus': 'Control whiteflies and remove infected plants.'
    },

    "cotton": {
        'aphids': 'Spray insecticides and encourage natural predators.',
        'army worm': 'Handpick larvae early and apply biological or chemical control.',
        'bacterial blight': 'Use resistant varieties and avoid overhead irrigation.',
        'cotton boll rot': 'Ensure proper drainage and use fungicides.',
        'green cotton boll': 'Harvest timely and avoid prolonged field exposure.',
        'healthy': 'No problems detected — maintain optimal growth conditions.',
        'powdery mildew': 'Use resistant varieties and apply sulfur-based fungicides.',
        'target spot': 'Remove infected leaves and apply fungicides.'
    }
}


def img_prediction(crop_name, image_path):

    interpreter = tf.lite.Interpreter(model_path=os.path.join("models", f"{crop_name}_model.tflite"))
    interpreter.allocate_tensors()

    # ==== Get input and output details ====
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    IMG_SIZE = (224, 224)
    

    def labels(index):
        
        rice_class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 
                    'Leaf Scald', 'Narrow Brown Leaf', 'Neck Blast', 'Rice Hispa', 
                    'Sheath Blight', 'tungro']
        wheat_class_names = ['Aphid','Black Rust','Blast', 'Brown Rust', 'Common Root Rot', 'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite', 'Septoria', 'Smut', 'Stem Fly', 'Tan Spot', 'Yellow Rust']    
        tomato_class_names = ['bacterial spot', 'early blight', 'healthy', 'late blight','leaf mold', 'septoria leaf spot', 'spider mites', 'target spot', 'mosaic virus', 'leaf curl virus'] 
        cotton_class_names = ['aphids', 'army worm', 'bacterial blight', 'cotton boll rot','green cotton boll', 'healthy', 'powdery mildew', 'target spot'] 

        labels_dict = {
            'rice': rice_class_names,
            'wheat': wheat_class_names,
            'tomato': tomato_class_names,
            'cotton': cotton_class_names
        }
        class_labels = labels_dict.get(crop_name, [])
        return class_labels[index]
    
    # ==== 4. Load and preprocess the image ====
    img_path = image_path  # Replace with your test image path
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
    predicted_label = labels(predicted_index)
    confidence = np.max(output_data) * 100

    solution = disease_solutions[crop_name][predicted_label]

    return {
        "label": f"{crop_name} class: {predicted_label}",
        "confidence": f"{confidence:.2f}",
        "solution": solution
    }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/crops')
def crops():
    return render_template('crops.html')

@app.route('/<crop_name>', methods=['GET', 'POST'])
def crop_page(crop_name):
    crop_name = crop_name.lower()
    if crop_name not in ['rice', 'wheat', 'tomato', 'cotton']:
        return redirect(url_for('crops'))

    uploaded_image_url = None
    prediction_result = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filepath = os.path.join(STATIC_FOLDER_user_files, file.filename)
            file.save(filepath)

            prediction_result = img_prediction(crop_name, filepath)
            uploaded_image_url = f"/static/user_files/{file.filename}"


    return render_template(
        f"{crop_name}.html",
        uploaded_image=uploaded_image_url,
        prediction_result=prediction_result
    )

def on_closed():
    import signal

    for f in os.listdir("static/user_files"):
        os.remove(os.path.join("static/user_files", f))
        
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    window = webview.create_window("Crop Disease Detection", app)
    window.events.closed += on_closed
    webview.start()
