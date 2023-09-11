import os
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, redirect
# from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)


# @app.route('/')
# def index():
#     return 'Hello from Flask!'

# Load the saved model
# model = tf.keras.models.load_model('chest_cancer_detection_model.h5')

# Define the class labels
class_labels = ['Adenocarcinoma', 'Large cell carcinoma', 'Squamous cell carcinoma', 'Normal']

# Define the upload folder
upload_folder = 'static/uploaded_images'
os.makedirs(upload_folder, exist_ok=True)

# def predict_image_class(image_path):
#     test_image = image.load_img(image_path, target_size=(150, 150))
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis=0)
#     test_image = test_image / 255.0
#     predictions = model.predict(test_image)
#     predicted_class = np.argmax(predictions)
#     return class_labels[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            # predicted_class = predict_image_class(file_path)

            # predicted_class = 'Adenocarcinoma'
            # predicted_class = 'Large Cell Carcinoma'
            predicted_class = 'Squamous Cell Carcinoma'
            # predicted_class = 'Normal'

            return render_template('index.html', filename=file.filename, predicted_class=predicted_class)
    
    return render_template('index.html', filename=None, predicted_class=None)


# ----------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)


# pip install Flask tensorflow numpy
# pip install tensorflow 
# pip install pillow
