import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from cv2 import reduce
from django.shortcuts import resolve_url
from importlib_metadata import method_cache
from keras.preprocessing import image 
import pandas as pd
import os
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
# from keras.preprocessing import image

from flask import Flask, render_template, request,redirect,url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import numpy as np

from keras.models import load_model
model = load_model("D:/capstone/archive/chest_xray/classifier_k.h5")

app = Flask(__name__)

def model_predict(img_path,model):
    test_image = image.load_img(img_path,target_size=(150,150))
    test_image = image.img_to_array(test_image)
    test_image = test_image.reshape(-1,150,150,1)
    # test_image = np.expand_dims(test_image, axis = 0)
    result = (model.predict(test_image)>0.5).astype("int32")
    print(result)
    if result[0] == 1:
        return "Pneunomia"
    else:
        return 'Normal'

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'Test',secure_filename(f.filename))
        print(file_path)
        preds = model_predict(file_path,model)
        return preds
    return None
    
if __name__ == "__main__":
    app.run(debug=True)