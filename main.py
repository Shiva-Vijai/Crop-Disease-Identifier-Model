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
        'Bacterial Leaf Blight': {
            "cause": "Caused by Xanthomonas oryzae pv. oryzae; spreads through infected seeds, rain splash, and standing water under humid conditions.",
            "solution": "Use resistant varieties (e.g., IR64, ADT38), maintain good drainage, and spray Streptomycin 100 ppm + Copper oxychloride 0.3% at early infection stage."
        },
        'Brown Spot': {
            "cause": "Caused by Bipolaris oryzae fungus, favored by nutrient deficiency and humid weather.",
            "solution": "Apply balanced fertilizers (especially nitrogen and potassium), maintain proper water level, and spray Mancozeb 0.25% or Carbendazim 0.1%."
        },
        'Healthy': {
            "cause": "No signs of infection or nutrient stress.",
            "solution": "Maintain good irrigation, balanced fertilizer use, and regular monitoring for pests and diseases."
        },
        'Leaf Blast': {
            "cause": "Caused by Magnaporthe oryzae fungus; favored by excess nitrogen and high humidity.",
            "solution": "Use resistant varieties (CO 43, ADT 44), avoid excess nitrogen, and spray Tricyclazole 0.06% or Isoprothiolane 0.1%."
        },
        'Leaf Scald': {
            "cause": "Caused by Microdochium oryzae under poor drainage and humid conditions.",
            "solution": "Improve drainage, remove infected residues, and apply Propiconazole 0.1% if infection spreads."
        },
        'Narrow Brown Leaf': {
            "cause": "Caused by Cercospora oryzae, often associated with potassium deficiency and high humidity.",
            "solution": "Apply potassium fertilizer, maintain field hygiene, and spray Mancozeb 0.25% if severe."
        },
        'Neck Blast': {
            "cause": "Same fungus as leaf blast (Magnaporthe oryzae), attacks panicle neck during flowering.",
            "solution": "Apply Tricyclazole 0.06% or Edifenphos 0.1% at booting stage and repeat after 10 days."
        },
        'Rice Hispa': {
            "cause": "Caused by Dicladispa armigera beetle; adults scrape leaves and larvae feed within them.",
            "solution": "Handpick beetles early, avoid over-fertilization, and spray Chlorpyrifos 0.05% or Quinalphos 0.05%."
        },
        'Sheath Blight': {
            "cause": "Caused by Rhizoctonia solani fungus in dense and nitrogen-rich fields.",
            "solution": "Reduce plant density, manage nitrogen, and apply Validamycin 0.25% or Hexaconazole 0.1% at early infection."
        },
        'tungro': {
            "cause": "Caused by rice tungro virus transmitted by green leafhopper (Nephotettix virescens).",
            "solution": "Use resistant varieties (ASD 16), remove infected plants, and control leafhopper using Imidacloprid 0.05%."
        }
    },

    "wheat": {
        'Aphid': {
            "cause": "Infestation by Sitobion avenae aphids sucking sap during grain filling stage.",
            "solution": "Spray Imidacloprid 0.03% or Thiamethoxam 0.025%, and encourage natural predators like ladybird beetles."
        },
        'Black Rust': {
            "cause": "Caused by Puccinia graminis fungus; spreads via wind-borne spores in warm, moist conditions.",
            "solution": "Plant resistant varieties (HD 2967), and spray Propiconazole 0.1% or Mancozeb 0.25% at early symptoms."
        },
        'Blast': {
            "cause": "Caused by Magnaporthe oryzae (wheat pathotype), prevalent in humid regions with delayed sowing.",
            "solution": "Avoid late planting, ensure proper drainage, and spray Tricyclazole 0.06% at heading stage."
        },
        'Brown Rust': {
            "cause": "Caused by Puccinia recondita fungus; develops in cool, moist weather.",
            "solution": "Grow resistant varieties, remove volunteer wheat plants, and apply Propiconazole 0.1%."
        },
        'Common Root Rot': {
            "cause": "Caused by Bipolaris sorokiniana and Fusarium species in dry or compacted soils.",
            "solution": "Rotate crops, use well-drained soil, and treat seeds with Thiram 2.5 g/kg before sowing."
        },
        'Fusarium Head Blight': {
            "cause": "Caused by Fusarium graminearum under high humidity during flowering.",
            "solution": "Spray Carbendazim 0.1% at flowering, avoid overhead irrigation, and rotate with non-cereal crops."
        },
        'Healthy': {
            "cause": "No visible pest or disease symptoms.",
            "solution": "Continue proper irrigation, fertilization, and weed control."
        },
        'Leaf Blight': {
            "cause": "Caused by Bipolaris sorokiniana; worsens with late sowing and nutrient deficiency.",
            "solution": "Apply Mancozeb 0.25%, improve nitrogen management, and use clean seed."
        },
        'Mildew': {
            "cause": "Caused by Blumeria graminis; thrives in cool, humid climates.",
            "solution": "Ensure air circulation and spray Wettable Sulfur 0.3% or Hexaconazole 0.1%."
        },
        'Mite': {
            "cause": "Caused by Aceria tosichella mites during hot, dry weather.",
            "solution": "Apply Sulfur dust 20–25 kg/ha and ensure adequate irrigation."
        },
        'Septoria': {
            "cause": "Caused by Septoria tritici fungus, infecting leaves during prolonged wetness.",
            "solution": "Remove crop residues, rotate crops, and spray Propiconazole 0.1% or Mancozeb 0.25%."
        },
        'Smut': {
            "cause": "Seed-borne disease caused by Ustilago tritici fungus.",
            "solution": "Treat seeds with Carboxin 2.5 g/kg or Tebuconazole 1 g/kg before planting."
        },
        'Stem Fly': {
            "cause": "Caused by Atherigona naqvii fly; larvae bore into wheat stems.",
            "solution": "Spray Dimethoate 0.05% at early infestation and avoid staggered sowing."
        },
        'Tan Spot': {
            "cause": "Caused by Pyrenophora tritici-repentis fungus surviving on debris.",
            "solution": "Remove debris, rotate crops, and spray Mancozeb 0.25% at first sign."
        },
        'Yellow Rust': {
            "cause": "Caused by Puccinia striiformis fungus; common in cool, moist conditions.",
            "solution": "Grow resistant varieties (HD 3086) and spray Propiconazole 0.1% or Mancozeb 0.25%."
        }
    },

    "tomato": {
        'bacterial spot': {
            "cause": "Caused by Xanthomonas campestris pv. vesicatoria; spreads through splashing water and contaminated seeds.",
            "solution": "Remove infected leaves, avoid overhead irrigation, and spray Copper oxychloride 0.3% or Streptocycline 100 ppm."
        },
        'early blight': {
            "cause": "Caused by Alternaria solani fungus, common in warm and humid climates.",
            "solution": "Rotate crops, prune lower leaves, and spray Mancozeb 0.25% or Chlorothalonil 0.2%."
        },
        'healthy': {
            "cause": "No signs of stress or infection.",
            "solution": "Maintain optimal nutrition, irrigation, and regular pest monitoring."
        },
        'late blight': {
            "cause": "Caused by Phytophthora infestans, spreads in cool, moist environments.",
            "solution": "Remove affected plants, avoid overhead watering, and spray Metalaxyl + Mancozeb 0.25%."
        },
        'leaf mold': {
            "cause": "Caused by Fulvia fulva fungus, often in humid greenhouses.",
            "solution": "Improve ventilation, reduce humidity, and spray Chlorothalonil 0.2% or Copper oxychloride 0.3%."
        },
        'septoria leaf spot': {
            "cause": "Caused by Septoria lycopersici fungus; spreads by rain splash and tools.",
            "solution": "Remove infected leaves and spray Mancozeb 0.25% weekly until control."
        },
        'spider mites': {
            "cause": "Caused by Tetranychus urticae mites in hot, dry weather.",
            "solution": "Spray Abamectin 0.002% or Dicofol 0.05%, and keep humidity up."
        },
        'target spot': {
            "cause": "Caused by Corynespora cassiicola fungus under high humidity.",
            "solution": "Remove infected foliage and spray Mancozeb 0.25% or Copper oxychloride 0.3%."
        },
        'mosaic virus': {
            "cause": "Caused by Tobacco mosaic virus, spreads via contact or insect vectors.",
            "solution": "Remove infected plants, disinfect tools, and control aphids/whiteflies."
        },
        'leaf curl virus': {
            "cause": "Caused by Tomato leaf curl virus transmitted by whiteflies.",
            "solution": "Spray Imidacloprid 0.03%, remove infected plants, and use resistant hybrids."
        }
    },

    "cotton": {
        'aphids': {
            "cause": "Caused by Aphis gossypii feeding on sap, especially during early stages.",
            "solution": "Spray Imidacloprid 0.03% or Thiamethoxam 0.025% and conserve ladybird beetles."
        },
        'army worm': {
            "cause": "Caused by Spodoptera litura larvae feeding on leaves and bolls.",
            "solution": "Handpick larvae early, install pheromone traps, and spray Emamectin Benzoate 0.002%."
        },
        'bacterial blight': {
            "cause": "Caused by Xanthomonas citri pv. malvacearum, spreads via rain and infected seed.",
            "solution": "Use resistant varieties, avoid overhead irrigation, and treat seeds with Copper oxychloride 0.3%."
        },
        'cotton boll rot': {
            "cause": "Caused by fungi like Rhizopus and Fusarium in humid, poorly drained fields.",
            "solution": "Ensure proper drainage, spray Carbendazim 0.1%, and harvest timely."
        },
        'green cotton boll': {
            "cause": "Due to delayed harvest or bollworm damage causing unopening bolls.",
            "solution": "Harvest timely, control bollworms, and maintain uniform irrigation."
        },
        'healthy': {
            "cause": "No pest or fungal presence.",
            "solution": "Continue optimal field conditions and pest monitoring."
        },
        'powdery mildew': {
            "cause": "Caused by Leveillula taurica fungus; thrives in dry, cool conditions.",
            "solution": "Spray Wettable Sulfur 0.3% or Hexaconazole 0.1%, and avoid dense planting."
        },
        'target spot': {
            "cause": "Caused by Corynespora cassiicola fungus under humid weather.",
            "solution": "Remove infected leaves and apply Mancozeb 0.25% or Carbendazim 0.1%."
        }
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

    prediction_result = {
    "cause": disease_solutions[crop_name][predicted_label]["cause"],
    "solution": disease_solutions[crop_name][predicted_label]["solution"]
}
    solution = disease_solutions[crop_name][predicted_label]["solution"]
    cause= disease_solutions[crop_name][predicted_label]["cause"]

    return {
        "label": f"{crop_name} class: {predicted_label}",
        "confidence": f"{confidence:.2f}",
        "solution": solution,
        'cause': cause,
        "prediction_result":prediction_result
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
