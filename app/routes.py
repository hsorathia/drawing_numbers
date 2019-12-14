from load import *
from flask import Flask, render_template, flash, redirect, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from flask import current_app as app
from . import db
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import LoginForm, RegisterForm
from app.models import User
from werkzeug.urls import url_parse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import base64
import sys
import os
import pickle
import neural

sys.path.append(os.path.abspath("./model"))


@app.route('/', methods=['GET', 'POST'])
def home():
    form_login = LoginForm()
    form_register = RegisterForm()
    return render_template('landing.html', form_login=form_login, form_register=form_register)


@app.route('/draw', methods=['GET', 'POST'])
def draw():
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('draw.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
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
    def prepImage():
        """
        Turns image from png to a 2d grayscale array
        """
        # open image
        img = Image.open("result.png", 'r')
        # img.show()
        # resize image
        img = img.resize((28, 28), Image.ANTIALIAS)
        # make grayscale
        img = img.convert('L')
        # img.show()
        arr = np.array(img)
        # print(arr)
        return arr

    ### retrieve data
    # load sizes
    fsizes = open('nn_sizes.pkl', 'rb')
    sizes = pickle.load(fsizes)
    fsizes.close()

    # load biases
    fbiases = open('nn_biases.pkl', 'rb')
    biases = pickle.load(fbiases)
    fbiases.close()

    # load weights
    fweights = open('nn_weights.pkl', 'rb')
    weights = pickle.load(fweights)
    fweights.close()

    # set up neural network based on file
    net = neural.Network(sizes=sizes)
    net.loadNodes(biases=biases, weights=weights)

    # running neural network
    arr = prepImage()
    # convert arr to 1d
    arr = np.reshape(arr, 784)
    arr = [(255-x)/256 for x in arr]
    result = (net.feedforward(arr))
    [print(i, "|", x) for i,x in enumerate(result)]
    my_guess = np.argmax(result)
    print ("result:", np.argmax(result))

    return render_template('draw.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        print(request.form["username"])
        user = User.query.filter_by(username=request.form["username"]).first()
        if user is None or not user.check_password(request.form["password"]):
            flash('Invalid username or password')
            return redirect(url_for('home'))
        login_user(user)
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        print(request.form["password"])
        user = User(
            username=request.form["username"], email=request.form["email"])
        user.set_password(request.form["password"])
        db.create_all()
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('home'))
    return redirect(url_for('home'))


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    user = current_user
    return render_template('profile.html', username=user.username)


def parseImg(imageData):
    # parse canvas image bytes and save as result.png
    imageString = re.search(b'base64,(.*)', imageData).group(1)
    with open('result.png', 'wb') as result:
        result.write(base64.decodebytes(imageString))
