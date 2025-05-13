import tensorflow as tf
import numpy as np
import os
from PIL import Image
import json
from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
from flask import send_file
# from xhtml2pdf import pisa
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using session

# Load model
model = tf.keras.models.load_model("fusion_model.h5")

# Load pesticide recommendations
with open(r"C:\Users\nayak\Documents\GitHub\Mini-Project\disease_pesticide_recommendations.json") as f:
    pesticide_data = json.load(f)

# Class index mapping
class_mapping = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Prediction function
def model_predict(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    disease = class_mapping[class_index]
    pesticide = pesticide_data.get(disease, "No recommendation available.")
    return disease, pesticide

# Home and prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        path = os.path.join("static", image.filename)
        image.save(path)

        disease, pesticide = model_predict(path)

        # Save prediction to session history
        if 'history' not in session:
            session['history'] = []
        session['history'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'disease': disease,
            'pesticide': pesticide,
            'image': path
        })
        session.modified = True

        return render_template("result.html", image_path=path, prediction=disease, pesticide=pesticide)
    
    return render_template("index.html")

# Disease history dashboard
@app.route('/history')
def history():
    history_data = session.get('history', [])
    return render_template('history.html', history=history_data)


@app.route('/download_report', methods=['POST'])
def download_report():
    disease = request.form.get('disease')
    pesticide = request.form.get('pesticide')
    image_path = request.form.get('image_path')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = render_template("report_template.html", 
                           disease=disease, 
                           pesticide=pesticide, 
                           timestamp=timestamp, 
                           image_path=image_path)

    pdf = BytesIO()
    pisa.CreatePDF(html, dest=pdf)
    pdf.seek(0)
    return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name='report.pdf')


if __name__ == "__main__":
    app.run(debug=True)
