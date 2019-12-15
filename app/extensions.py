from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_heroku import Heroku

db = SQLAlchemy()
login = LoginManager()
heroku = Heroku()