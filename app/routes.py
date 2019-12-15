from load import *
from flask import Flask, render_template, flash, redirect, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from flask import current_app as app
from . import db
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import LoginForm, RegisterForm
from app.models import User, UserNumbers
from werkzeug.urls import url_parse
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import base64
import sys
import os
import pickle
import neural

sys.path.append(os.path.abspath("./model"))
tempGuess = None

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Landing page that describes the application
    """
    form_login = LoginForm()
    form_register = RegisterForm()
    return render_template('landing.html', form_login=form_login, form_register=form_register)


@app.route('/draw', methods=['GET', 'POST'])
def draw():
    """
    Drawing page that allows users to draw/save numbers
    """
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('draw.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    """
    Data route used to send & save images from frontend to backend
    """
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    # gets canvas image
    parseImg(request.get_data())
    # print(request.get_data())
    # read parsed image
    # d = Image.open('result.png', 'r')
    # resize image to 28x28
    # d = d.resize((20,20))
    # d = d.resize((28, 28), Image.ANTIALIAS)
    # d.save('result.png')

    # img = mpimg.imread('result.png')
    # print(img)
    # imgplot = plt.imshow(img)
    # plt.show()
    # d = np.asarray(d)
    # # reshape the data to feed into NN
    # d = d.reshape(-1, 1, 28, 28)

    def prepImage():
        """
        Turns image from png to a 2d grayscale array
        """
        # open image
        img = Image.open("result.png", 'r')
        # img.show()
        # resize image
        img = img.resize((20, 20))
        img = img.resize((28, 28), Image.ANTIALIAS)
        # enhancer = ImageEnhance.Sharpness(img)
        # img = enhancer.enhance(0.5)
        # make grayscale
        img = img.convert('L')
        # disp = plt.imshow(img)
        # plt.show()
        # img.show()
        arr = np.array(img)
        # print(arr)
        return arr

    # retrieve data
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
    arr = np.reshape(arr, (784, 1))
    arr = (255-arr)/256

    # print(arr)
    result = (net.feedforward(arr))
    # [print(i, "|", x) for i, x in enumerate(result)]
    my_guess = np.argmax(result)
    print("result:", my_guess)

    usernumber = UserNumbers(
        userID=current_user.id, image=request.get_data(), guess=int(my_guess))
    db.create_all()
    db.session.add(usernumber)
    db.session.commit()
    print(usernumber)
    global tempGuess
    tempGuess = int(my_guess)
    return render_template('draw.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login route for users to log into their account
    """
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        print(request.form["username"])
        user = User.query.filter_by(username=request.form["username"]).first()
        if user is None or not user.check_password(request.form["password"]):
            flash('Invalid username or password')
            return redirect(url_for('home'))
        login_user(user)
        return redirect(url_for('draw'))
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Register route for users to register a new account
    """
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        print(request.form["password"])
        user = User(
            username=request.form["username"], email=request.form["email"])
        user.set_password(request.form["password"])
        if User.query.filter_by(username=request.form['username']).first():
            flash('Username already taken')
        else:
            db.create_all()
            db.session.add(user)
            db.session.commit()
            flash('Congratulations, you are now a registered user!')
            # login_user(user)
            # return redirect(url_for('draw'))
    return redirect(url_for('home'))


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """
    Logout route for users to log out of their account
    """
    logout_user()
    return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    """
    Profile page that displays the user's saved numbers, along with the neural network's guess
    """
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    user = current_user
    usernumbers = user.usernumbers.all()
    final = []
    for numbers in usernumbers:
        information = []
        information.append(numbers.image.decode())
        information.append(numbers.guess)
        information.append(numbers.id)
        #print(information)
        final.append(information)
    return render_template('profile.html', username=user.username, usernumbers=final)


@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    print(id)
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    userNums = UserNumbers.query.filter_by(id=id).first()
    db.create_all()
    db.session.delete(userNums)
    db.session.commit()

@app.route('/get', methods=['GET', 'POST'])
def get_data():
    if request.method == "GET":
        return str(tempGuess)

def parseImg(imageData):
    """
    Helper function used to parse base64 into an image
    """
    if not current_user.is_authenticated:
        return redirect(url_for('home'))
    # parse canvas image bytes and save as result.png
    imageString = re.search(b'base64,(.*)', imageData).group(1)
    with open('result.png', 'wb') as result:
        result.write(base64.decodebytes(imageString))
