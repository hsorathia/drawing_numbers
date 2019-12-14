from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from .extensions import db, login



def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)
    app.debug = True
    db.init_app(app)
    login.init_app(app)
    login.loginview = 'login'

    with app.app_context():
        from . import routes, models, forms
        db.create_all()
        return app
