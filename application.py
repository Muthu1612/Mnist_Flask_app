from flask import Flask, render_template, request, jsonify, url_for, send_from_directory 
import numpy as np
import keras.models
from keras.models import model_from_json
import re
import base64
from PIL import Image 
import sys 
import os
# sys.path.append(os.path.abspath("./model"))
import cv2

# import imageio
# from PIL import Image
# from scipy.misc import imresize
# from load import *
import numpy as np
application = Flask(__name__)
global model, graph
# model, graph = init()
import json 
@application.route('/')
def index():


    
    return render_template("index.html")

@application.route('/predict/', methods=['POST', 'GET'])
def predict():
    # if not request.script_root:
    #     # this assumes that the 'index' view function handles the path '/'
    #     request.script_root = url_for('index', _external=True)
    parseImage(request.get_data())
    # x = imageio.imread('output.png', pilmode='L')
    # x = imresize(x, (28, 28))
    # # x = x.reshape(28, 28, 1)
    # x = np.invert(x)
    x = cv2.imread('output.png')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.invert(x)
    x = cv2.resize(x,(28,28))



    x=np.array(x)

    x = x.reshape(1,28,28,1)
    x=x/255
    # with graph.as_default():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("weights.h5")
    print("Loaded Model from disk")

    loaded_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    out = loaded_model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    response = np.array_str(np.argmax(out, axis=1))
    return response 

        
def parseImage(imgData):

    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.b64decode(imgstr))

@application.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(application.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
# if __name__ == '__main__':
#     application.debug = True
#     # port = int(os.environ.get("PORT", 8000))

#     application.run(host='0.0.0.0', port=port)
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8000, debug=True)