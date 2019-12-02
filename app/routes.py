from flask import render_template, flash, redirect, url_for
from flask import request
from werkzeug.urls import url_parse
from flask import Flask           # import flask

app = Flask(__name__)             # create an app instance


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == "__main__":        # on running python app.py
    app.run(debug=True)
