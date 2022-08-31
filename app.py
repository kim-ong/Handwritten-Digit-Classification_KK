import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
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
    predictions = ()
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    inverted = np.invert(my_image)
    my_image_re = resize(inverted, (32,32,1))
    model.run_eagerly=True  
    probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
    print(probabilities)
    #Step 3
    number_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index = np.argsort(probabilities)
    if probabilities[index[9]] > 0.9:
      grade = "Good Job!"
    else:
      grade = "Learn to write this number in Chatbot."
    predictions = {
      "digit":number_to_class[index[9]],
      "prob":probabilities[index[9]],
      "comment":grade
     }
    #Step 5
    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
