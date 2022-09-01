import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

import tensorflow as tf 
from tensorflow.keras.utils import load_img

from keras.models import load_model 
from keras.backend import set_session

import matplotlib.pyplot as plt 
import numpy as np

print("Loading model") 
global model 
model = load_model('handwriting.h5') 

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    myfile = os.path.join('uploads', filename)
    img = load_img(myfile, color_mode="grayscale", target_size=(32, 32))
    img = np.invert(img)
    img = img.reshape(1, 32, 32, 1)
    img = np.array(img)
    img = img.astype('float32')
    img /= 255
    probabilities = model.predict(img)[0,:]
    print(probabilities)
    number_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index = np.argsort(probabilities)
    if probabilities[index[9]] > 0.9:
     grade = "Good Job!"
    else:
     grade = "Try again!"
    predictions = {
      "digit":number_to_class[index[9]],
      "prob" :probabilities[index[9]],
      "comment":grade
     }
    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
