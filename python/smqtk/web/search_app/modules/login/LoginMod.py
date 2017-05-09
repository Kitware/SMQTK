
import flask
import functools
import json
import os.path

import requests

from smqtk.utils import SmqtkObject
from smqtk.utils.url import url_join


script_dir = os.path.dirname(os.path.abspath(__file__))


# noinspection PyUnusedLocal
class LoginMod (SmqtkObject, flask.Blueprint):

    def __init__(self, name, parent_app, url_prefix=None):
        """
        Initialize the login module

        :param parent_app: Parent application that is loading this using this
            module.
        :type parent_app: flask.Flask

        """
        super(LoginMod, self).__init__(
            name, __name__,
            template_folder=os.path.join(script_dir, 'templates'),
            url_prefix=url_prefix
        )

        # Pull in users configuration file from etc directory
        users_config_file = os.path.join(script_dir, 'users.json')
        if not os.path.isfile(users_config_file):
            raise RuntimeError("Couldn't find expected users config file -> %s"
                               % users_config_file)
        with open(users_config_file) as infile:
            self.__users = json.load(infile)

        #
        # Routing
        #
        @self.route('/login', methods=['get'])
        def login():
            # Render the login page, then continue on to a previously requested
            # page, or the home page.
            # noinspection PyUnresolvedReferences
            return flask.render_template(
                'login.html',
                next=flask.request.args.get('next', '/'),
                username=flask.request.args.get('username', ''),
            )

        @self.route('/login.passwd', methods=['post'])
        def login_password():
            """
            Log-in processing method called when submitting form displayed by
            the login() method above.

            There will always be a 'next' value defined in the form (see above
            method).

            """
            userid = flask.request.form['login']
            next_page = flask.request.form['next']

            if userid in self.__users:
                # Load user
                user = self.__users[userid]
                self._log.debug("User info selected: %s", user)

                if flask.request.form['passwd'] != user['passwd']:
                    flask.flash("Authentication error for user id: %s" % userid,
                                "error")
                    return flask.redirect("/login?next=%s&username=%s"
                                          % (next_page, userid))

                flask.flash("Loading user: %s" % userid, "success")
                self._login_user(userid, user)
                return flask.redirect(next_page)
            else:
                flask.flash("Unknown user: %s" % userid, 'error')
                return flask.redirect("/login?next=%s" % next_page)

        @self.route('/logout')
        def logout():
            """
            Logout the current user
            """
            if 'user' in flask.session:
                del flask.session['user']
            else:
                flask.flash("No user currently logged in!", "error")
            return flask.redirect("/")

    #
    # Utility methods
    #

    @staticmethod
    def _login_user(userid, user_info):
        """
        "log-in" the user in the current session. This adds the name and role
        list to the session. Only one user logged in at a time.

        :param userid: String ID of the user
        :type userid: str

        :param user_info: The user dictionary as recorded in our users.json
            config file.
        :type user_info: dict of (str, str or list of str)

        """
        flask.session['user'] = {
            'id': userid,
            'fullname': user_info['fullname'],
            'roles': user_info['roles']
        }

    @staticmethod
    def login_required(f):
        """
        Decorator for URLs that require login.

        Girder aware. Login-required URLs can be passed:

            "girder_token"
            "girder_origin"
            "girder_apiRoot"

        argument,  We will check the given token with the origin URL and API
        root.

        :param f: Function to be wrapped.

        """
        log = LoginMod.get_logger()

        @functools.wraps(f)
        def deco(*args, **kwds):
            # Combine to handle both GET arguments and form data
            c = {}
            c.update(dict(flask.request.form.iteritems()))
            c.update(dict(flask.request.args.iteritems()))
            log.debug("Combined arguments: %s", c)

            if not {'girder_token', 'girder_origin', 'girder_apiRoot'} \
                    .difference(c.keys()):
                g_token = c['girder_token']
                log.debug("G-token: %s", g_token)
                g_origin = c['girder_origin']
                log.debug("G-origin: %s", g_origin)
                g_apiRoot = c['girder_apiRoot']
                log.debug("G-apiRoot: %s", g_apiRoot)

                g_api_header = {'Girder-Token': g_token}

                # Attempt getting current user using supplied token. If it
                # succeeds and matches the given user, log them in here.

                # Get user ID from token
                log.debug('Getting Girder current token info')
                r = requests.get(url_join(g_origin, g_apiRoot, 'token/current'),
                                 headers=g_api_header)
                if r.json() is None:
                    flask.flash("Invalid Girder token credentials token",
                                'error')
                    return flask.redirect(flask.url_for("login.login")
                                          + "?next=" + flask.request.url)
                else:
                    g_userId = r.json()['userId']

                    # Get user label and name from ID
                    log.debug("Getting user info from ID")
                    r = requests.get(url_join(g_origin, g_apiRoot, 'user',
                                              g_userId),
                                     headers=g_api_header)
                    user_label = r.json()['login']
                    user_fullname = ' '.join([r.json()['firstName'],
                                              r.json()['lastName']])

                    log.debug("Logging user '%s' in", user_fullname)
                    # Arbitrarily log the user in as a guest
                    LoginMod._login_user(user_label, {
                        'fullname': user_fullname,
                        'roles': ['guest'],
                    })

            if 'user' not in flask.session:
                flask.flash("Login required!", 'error')
                return flask.redirect(flask.url_for("login.login")
                                      + "?next=" + flask.request.url)
            else:
                # TODO: Check that user has permission, else redirect
                return f(*args, **kwds)
        return deco
