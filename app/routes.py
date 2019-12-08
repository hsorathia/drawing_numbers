from load import *
from flask import Flask, render_template, flash, redirect, url_for, request
from flask import current_app as app
from werkzeug.urls import url_parse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import base64
import sys
import os
sys.path.append(os.path.abspath("./model"))

@app.route('/')
@app.route('/landing')
def home():
    return render_template('landing.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    # gets canvas image
    parseImg(request.get_data())
    # read parsed image
    d = Image.open('result.png', 'r')
    # resize image to 28x28
    d = d.resize((28, 28))
    d.save('result.png')
    img = mpimg.imread('result.png')
    imgplot = plt.imshow(img)
    plt.show()
    d = np.asarray(d)
    # reshape the data to feed into NN
    d = d.reshape(-1, 1, 28, 28)

    print(d)
    return render_template('index.html')


def parseImg(imageData):
    # parse canvas image bytes and save as result.png
    imageString = re.search(b'base64,(.*)', imageData).group(1)
    with open('result.png', 'wb') as result:
        result.write(base64.decodebytes(imageString))

