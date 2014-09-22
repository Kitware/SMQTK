"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import flask
from functools import wraps
from flask import Blueprint, redirect, render_template, request, session, flash, url_for, current_app
#from flask_openid import OpenID
#from flask_oauth import OAuth

mod = Blueprint('login', __name__)
#oid = OpenID()

# Load users
import os
thispath = os.path.dirname(os.path.abspath(__file__))
from WebUI import app
fin = open(os.path.join(app.config['ETC_DIR'], 'users.json'))
import json
USERS = json.loads(fin.read())

# Decorator for urls that require login
def login_required(f):
    """Checks whether user is logged in or redirects to login"""
    @wraps(f)
    def decorator(*args, **kwargs):
        if not 'user' in flask.session:
            flask.flash("Login required !", "error")
            return flask.redirect(url_for("login.login") + "?next=" + flask.request.url)
        else:
            return f(*args, **kwargs)
    return decorator

# Decorator for urls that require specific role
def role_required(role):
    """Checks whether user is logged in or redirects to login"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):

            if not 'user' in flask.session:
                flask.flash("this Login required !", "error")
                return flask.redirect(url_for("login.login") + "&next=" + flask.request.url)
            else:
                if 'roles' in flask.session["user"]:
                    if role in flask.session["user"]["roles"]:
                        # flask.flash("Found access for \"" + role + "\" group :(", "success")
                        return f(*args, **kwargs)
                flask.flash("Access restricted only to login group \"" + role + "\" group :(", "error")
                return flask.redirect(url_for("home"))
        return decorated_function
    return decorator

@mod.route('/login', methods=["get"])
def login():
    return render_template("login.html", next=flask.request.args.get("next","/home"))

@mod.route('/login.passwd', methods=['post'])
def login_passwd():
    # Try to find the user
    userid = request.form["login"]
    app = flask.current_app
    if userid in USERS:
        # Load user
        user = USERS[userid]

        if user["passwd"] != request.form['passwd']:
            flash('Authentication Error for: ' + userid, "error")
            return redirect('/login')

        flask.flash("Loading user: "+userid, "success")
        return do_user_login(user, next=flask.request.form["next"])

    else:
        flash('Unknown user: ' + request.form['login'], "error")
        return redirect('/login')

def do_user_login(user, next="/home"):
    session['user'] = {
        'fullname': user["fullname"],
        'roles' : user["roles"],
        }

    flash('Successfully logged in user: ' + user["fullname"], 'success')
    return redirect(next)

@mod.route('/logout', methods=['GET', 'POST'])
def logout():
    """Does the login via OpenID. Has to call into `oid.try_login`
    to start the OpenID machinery.
    """
    # if we are already logged in, go back to were we came from
    flask.g.logged_in = False
    session.clear()

    return redirect(url_for('home'))

