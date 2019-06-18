# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#

import datetime
import glob
import logging
import os
from functools import wraps
from logging import Formatter, FileHandler

from authlib.flask.client import OAuth
from flask import Flask, render_template, request, jsonify, url_for, session, redirect
from rawapi import new_raw_client
from six.moves.urllib.parse import urlencode

# ----------------------------------------------------------------------------#
# App Config.
# ----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')
oauth = OAuth(app)

auth0 = oauth.register(
    'auth0',
    client_id=app.config['OAUTH_CLIENT_ID'],
    client_secret=app.config['OAUTH_CLIENT_SECRET'],
    api_base_url=app.config['OAUTH_API_BASE_URL'],
    access_token_url=app.config['OAUTH_ACCESS_TOKEN_URL'],
    authorize_url=app.config['OAUTH_AUTHORIZE_URL'],
    client_kwargs={
        'scope': 'openid profile',
    },
)

logging.basicConfig(
    level='INFO'
)


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'profile' not in session:
            # Redirect to Login page here
            return redirect('/login')
        return f(*args, **kwargs)

    return decorated


def create_client(session):
    if app.config['RAW_AUTH'] == 'auth0':
        return new_raw_client(
            executor_url=app.config['EXECUTOR_URL'],
            creds_url=app.config['CREDS_URL'],
            auth='auth0',
            auth0_auth_type='access_token',
            access_token=session['token_info']['access_token']
        )
    else:
        return new_raw_client()


def init_packages(session):
    """Initializes packages, s3 buckets, etc for this session"""
    client = create_client(session)
    # Registering buckets
    with open(os.path.join('raw_ini', 'buckets.txt')) as f:
        buckets = client.buckets_list()
        for line in f.readlines():
            values = line.split()
            if len(values) < 1:
                continue

            name = values[0]
            region = values[1] if len(values) >= 2 else None
            access_key = values[2] if len(values) >= 3 else None
            secret_key = values[3] if len(values) >= 4 else None

            if name not in buckets:
                app.logger.info('Registering bucket s3://%s' % name)
                client.buckets_register(name, region, access_key, secret_key)

    packages = client.packages_list_names()
    # Registering packages
    for filename in glob.glob(os.path.join('raw_ini', 'packages/*.rql')):
        name = os.path.basename(filename[:-4])
        if name in packages:
            app.logger.warning('overwriting package %s' % name)
            client.packages_delete(name)
        with open(filename) as f:
            app.logger.info('registering package %s' % name)
            client.packages_create(name, f.read())


# ----------------------------------------------------------------------------#
# Controllers.
# ----------------------------------------------------------------------------#
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}


@app.route('/')
def home():
    return redirect(url_for('diabetes'))


# Here we're using the /callback route.
@app.route('/callback')
def callback_handling():
    if app.config['RAW_AUTH'] == 'auth0':
        # Handles response from token endpoint
        # Stores token in flask session
        session['token_info'] = auth0.authorize_access_token()
        resp = auth0.get('userinfo')
        userinfo = resp.json()

        # Store the user information in flask session.
        session['jwt_payload'] = userinfo
        session['profile'] = {
            'user_id': userinfo['sub'],
            'name': userinfo['name'],
            'picture': userinfo['picture']
        }
    else:
        session['profile'] = {
            'user_id': 'local',
            'name': 'local user',
            'picture': ''
        }
    # initializes raw-client, buckets, etc.
    init_packages(session)
    return redirect(url_for('machines'))


@app.route('/login')
def login():
    if app.config['RAW_AUTH'] == 'auth0':
        return auth0.authorize_redirect(redirect_uri=url_for('callback_handling', _external=True),
                                        audience=app.config['OAUTH_AUDIENCE'])
    else:
        # Skips the auth0 login
        return redirect(url_for('callback_handling'))


@app.route('/logout')
def logout():
    return render_template('pages/logged_out.html')


@app.route('/do_logout')
def do_logout():
    # Clear session stored data
    session.clear()
    if app.config['RAW_AUTH'] == 'auth0':
        params = {'returnTo': url_for('logout', _external=True), 'client_id': app.config['OAUTH_CLIENT_ID']}
        return redirect(app.config['OAUTH_LOGOUT_URL'] + urlencode(params))
    else:
        return redirect(url_for('logout'))


@app.route('/diabetes/train')
@requires_auth
def diabetes_train():
    client = create_client(session)
    f1 = request.args.get('f1')
    f2 = request.args.get('f2')
    results = client.query('''
        predict := \python(x: collection(collection(double)), y: collection(double)): record(
                                                                                        prediction: mdarray(double, x), 
                                                                                        coef: mdarray(double, x), 
                                                                                        intercept: double
                                                                                    ) -> $$$
            import sklearn
            import sklearn.linear_model
            regr = sklearn.linear_model.LinearRegression()
            regr.fit(x, y)
            return dict(prediction=regr.predict(x), coef=regr.coef_, intercept=regr.intercept_)
        $$$;

        dataset := read("s3://raw-tutorial/ipython-demos/diabetes/diabetes_dataset.csv", cache := interval "1 hour");
        target := select Y from dataset;

        (
            xyz: (select {0} as x, {1} as y, Y as z from dataset),
            predict1: predict((select [{0}] from dataset), target),
            predict2: predict((select [{1}] from dataset), target),
            predict_both: predict((select [{0}, {1}] from dataset), target)
        )'''.format(f1, f2))
    return jsonify(results)


@app.route('/diabetes')
@requires_auth
def diabetes():
    return render_template('pages/diabetes.html')


# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    # db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

# ----------------------------------------------------------------------------#
# Launch.
# ----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
