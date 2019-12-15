from . import db
from . import login
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(128), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    usernumbers = db.relationship('UserNumbers', backref="author", lazy="dynamic")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)

class UserNumbers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userID = db.Column(db.Integer, db.ForeignKey('user.id'))
    image = db.Column(db.LargeBinary)
    values = db.Column(db.LargeBinary)
    guess = db.Column(db.Integer)
    

# neural network
# class NN(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     num_layers = db.Column(db.Integer)
#     biases = db.relationship('Bias', backref='Network')
#     weights = db.relationship('Weight', backref='Network')

# class Bias(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     value = db.Column(db.Integer)
#     net = db.Column(db.Integer, db.ForeignKey('nn.id'))

# class Weight(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     value = db.Column(db.Integer)
#     net = db.Column(db.Integer, db.ForeignKey('nn.id'))

@login.user_loader
def load_user(id):
    return User.query.get(int(id))
