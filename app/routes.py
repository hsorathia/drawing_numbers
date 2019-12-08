from load import *
from flask import Flask, render_template, flash, redirect, url_for, request
from flask import current_app as app
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('landing'))
    form = LoginForm()
    if form.validate_on_submit():
        # look at first result first()
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        # return to page before user got asked to login
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('landing')

        return redirect(next_page)
    return render_template('login.html', title='Sign in', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('landing'))
    form = RegisterForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

def parseImg(imageData):
    # parse canvas image bytes and save as result.png
    imageString = re.search(b'base64,(.*)', imageData).group(1)
    with open('result.png', 'wb') as result:
        result.write(base64.decodebytes(imageString))
