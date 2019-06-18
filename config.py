import os
from os import environ as env

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

# Enable debug mode.
DEBUG = True

# Secret key for session management. You can generate random strings here:
# https://randomkeygen.com/
SECRET_KEY = 'my precious'

SESSION_TYPE = 'filesystem'


AUTH0_DOMAIN = 'raw.eu.auth0.com'
OAUTH_CLIENT_ID = env['RAW_AUTH_ID']
OAUTH_CLIENT_SECRET = env['RAW_AUTH_SECRET']
OAUTH_API_BASE_URL = 'https://%s' % AUTH0_DOMAIN
OAUTH_ACCESS_TOKEN_URL = 'https://%s/oauth/token' % AUTH0_DOMAIN
OAUTH_AUTHORIZE_URL = 'https://%s/authorize' % AUTH0_DOMAIN
OAUTH_LOGOUT_URL = OAUTH_API_BASE_URL + '/v2/logout?'
OAUTH_AUDIENCE = 'https://just-ask.raw-labs.com'

EXECUTOR_URL='https://eu-just-ask.raw-labs.com/executor'
CREDS_URL='https://eu-just-ask.raw-labs.com/creds'
