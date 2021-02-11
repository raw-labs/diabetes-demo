# ----------------------------------------------------------------------------#
# Imports
# ----------------------------------------------------------------------------#
import datetime
import glob
import logging
import os
import subprocess
from logging import Formatter, FileHandler

from flask import Flask, render_template, request, jsonify, url_for, redirect
from rawapi import new_raw_client, RawException, Unauthorized

# ----------------------------------------------------------------------------#
# App Config.
# ----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

logging.basicConfig(
    level='INFO'
)


def get_client():
    return new_raw_client()


client = get_client()


def query(q):
    return client.query(q)


def read_values(f, properties):
    output = []
    for line in f.readlines():
        values = line.split()
        if len(values) < 1:
            continue
        d = dict()

        for n, name in enumerate(properties):
            if n < len(values):
                d[name] = values[n]
            else:
                d[name] = None
        output.append(d)
    return output


"""Initializes packages, s3 buckets, etc for this session"""
# Registering buckets
with open(os.path.join('raw_ini', 'buckets.txt')) as f:
    buckets = client.buckets_list()
    config = read_values(f, ["name", "region", "access_key", "secret_key"])
    for b in config:
        if b["name"] not in buckets:
            app.logger.info('Registering bucket s3://%s' % b["name"])
            client.buckets_register(b["name"], b["region"], b["access_key"], b["secret_key"])

try:
    with open(os.path.join('raw_ini', 'rdbms.txt')) as f:
        servers = client.rdbms_list()
        config = read_values(f, ["name", "type", "host", "port", "db", "user", "passwd"])
        for s in config:
            if s["name"] not in servers:
                if s["type"] == "postgresql":
                    client.rdbms_register_postgresql(s["name"], s["host"], s["db"], int(s["port"]), s["user"],
                                                     s["passwd"])
                elif s["type"] == "sqlserver":
                    client.rdbms_register_sqlserver(s["name"], s["host"], s["db"], int(s["port"]), s["user"],
                                                    s["passwd"])
                elif s["type"] == "oracle":
                    client.rdbms_register_oracle(s["name"], s["host"], s["db"], int(s["port"]), s["user"], s["passwd"])
                elif s["type"] == "mysql":
                    client.rdbms_register_mysql(s["name"], s["host"], s["db"], int(s["port"]), s["user"], s["passwd"])
                else:
                    app.logger.error('unsupported database type %s, skipping' % s["type"])
except FileNotFoundError:
    app.logger.info("no file found with dbms servers")

views = client.views_list_names()
print(views)
# creating views
files = glob.glob(os.path.join('raw_ini', 'views/*.rql'))
files.sort()
for filename in files:
    # view filenames are prepended with a number '01_' which specifies the order for creating the view
    # so removing first 3 characters and last 4 (file extension)
    name = os.path.basename(filename)[3:-4]
    if name not in views:
        with open(filename) as f:
            app.logger.info('creating view %s' % name)
            client.views_create(name, f.read())

packages = client.packages_list_names()
# Registering packages
for filename in glob.glob(os.path.join('raw_ini', 'packages/*.rql')):
    name = os.path.basename(filename[:-4])
    if name not in packages:
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


@app.route('/diabetes/train')
def diabetes_train():
    f1 = request.args.get('f1')
    f2 = request.args.get('f2')
    results = query('''
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
    app.run(debug=True, port=5010)

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
