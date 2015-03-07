
import flask
import functools
import json
import logging
import os.path


script_dir = os.path.dirname(os.path.abspath(__file__))


# noinspection PyUnusedLocal
class LoginMod (flask.Blueprint):

    def __init__(self, parent_app):
        """
        Initialize the login module

        :param parent_app: Parent application that is loading this using this
            module.
        :type parent_app: flask.Flask

        """
        super(LoginMod, self).__init__(
            'login', __name__,
            template_folder=os.path.join(script_dir, 'templates')
        )

        self.log = logging.getLogger('LoginMod')

        # Pull in users configuration file from etc directory
        users_config_file = os.path.join(parent_app.config['ETC_DIR'],
                                         'users.json')
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
            return flask.render_template(
                'login.html',
                next=flask.request.args.get('next', '/home'),
                username=flask.request.args.get('username', '')
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
                self.log.debug("User info selected: %s", user)

                if flask.request.form['passwd'] != user['passwd']:
                    flask.flash("Authentication error for user id: %s" % userid,
                                "error")
                    return flask.redirect("/login?next=%s&username=%s"
                                          % (next_page, userid))

                flask.flash("Loading user: %s" % userid, "success")
                self._login_user(user)
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
                return flask.redirect("/home")
            else:
                flask.flash("No user currently logged in!", "error")

    #
    # Utility methods
    #

    @staticmethod
    def _login_user(user):
        """
        "log-in" the user in the current session. This adds the name and role
        list to the session. Only one user logged in at a time.

        :param user: The user dictionary as recorded in our users.json config
            file.
        :type user: dict of (str, str or list of str)

        """
        flask.session['user'] = {
            'fullname': user['fullname'],
            'roles': user['roles']
        }

    @staticmethod
    def login_required(f):
        """
        Decorator for URLs that require login

        :param f: Function to be wrapped.

        """
        @functools.wraps(f)
        def deco(*args, **kwds):
            if not 'user' in flask.session:
                flask.flash("Login required!", 'error')
                return flask.redirect(flask.url_for("login.login")
                                      + "?next=" + flask.request.url)
            else:
                # TODO: Check that user has permission, else redirect
                return f(*args, **kwds)
        return deco
