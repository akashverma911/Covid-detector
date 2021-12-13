from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'covid_detector_model.h5'

model.load_model(MODEL_PATH)
 # model._make_predict_function()

def model_predict(img_path, model):
    target_size = (224,224)
    img = image.load_img(para_cell,target_size=target_size)
    my_image = image.img_to_array(img)
    my_image = np.expand_dims(my_image, axis=0)
    # my_image = preprocess_input(my_image, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
