import flask
from hashlib import sha1
import json
from mimetypes import guess_type
import os
import os.path
import requests
import re
import sys

from six.moves.urllib.parse import urlparse


RE_HTTP = re.compile(r'''^https?://.*''')


def log(s):
    print(s)
    sys.stdout.flush()


def proxy(new_location):
    log('  PROXYING TO %s' % new_location)
    error_message = None

    try:
        proxy_res = requests.get(new_location, timeout=(30.05, 30.05))

        res = flask.Response(proxy_res.iter_content(chunk_size=1024))
        res.status_code = proxy_res.status_code
        for k, v in proxy_res.headers.items():
            res.headers.add(k, v)

    except requests.exceptions.ConnectionError:
        error_message = 'downstream service not yet ready'

    except requests.exceptions.ReadTimeout:
        error_message = 'downstream service read timeout'

    if error_message is not None:
        res = flask.make_response(json.dumps({
            'message': error_message}))
        res.status_code = 503
        res.headers.add('content-type', 'application/json')

    # ``res`` either has a value from the try-catch or from the error catch.
    # noinspection PyUnboundLocalVariable
    return res


app = flask.Flask('wrapper0')


@app.route('/image/<img_hash>')
def image(img_hash):
    link_path = os.path.join('/links', img_hash)
    real_path = os.path.realpath(link_path)
    log('  SERVING IMAGE: ' + real_path)
    image_name = os.path.basename(real_path)

    def generate():
        with open(real_path, 'rb') as f:
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break
                yield chunk

    res = flask.Response(generate())
    c_type, encoding = guess_type(image_name)

    res.headers['content-type'] = c_type
    res.headers['content-encoding'] = encoding
    res.headers['content-disposition'] = 'inline'
    return res


@app.route('/<path:url_path>')
def catch_all0(url_path):
    query = flask.request.query_string.decode('utf8')
    # noinspection PyTypeChecker
    path = '?'.join(filter(bool, (url_path, query)))
    new_location = 'http://smqtk:12345/' + path
    return proxy(new_location)


@app.route('/nn/<path:url_path>')
@app.route('/nn/<int:n>/<path:url_path>')
@app.route('/nn/<int:n>/<int:start>/<path:url_path>')
@app.route('/nn/<int:n>/<int:start>/<int:end>/<path:url_path>')
def intercept_path(n=None, start=None, end=None, url_path=None):
    log('  INTERCEPTING')
    query = flask.request.query_string.decode('utf8')
    # noinspection PyTypeChecker
    url_path = '?'.join(filter(bool, (url_path, query)))

    uri = url_path
    if RE_HTTP.match(uri) is None:
        uri = 'http://' + uri

    log('  URI: ' + uri)
    parsed = urlparse(uri)
    if parsed.hostname == 'localhost':
        log('  REWRITING PATH HOSTNAME')
        log('    BEFORE: ' + url_path)

        scheme = str(parsed.scheme)
        hostname = 'wrapper'
        port = str(parsed.port)
        path = str(parsed.path)
        query = str(parsed.query)
        fragment = str(parsed.fragment)

        if not scheme:
            scheme = 'http://'

        if not scheme.endswith('://'):
            scheme += '://'

        if port and not port.startswith(':'):
            port = ':' + port

        if path and not path.startswith('/'):
            path = '/' + path

        if query and not query.startswith('?'):
            query = '?' + query

        if fragment and not fragment.startswith('#'):
            fragment = '#' + fragment

        uri = ''.join((scheme, hostname, port, path, query, fragment))
        log('    AFTER : ' + uri)

    new_location = ['nn']
    for value in (n, start, end):
        if value is None:
            break

        new_location.append(str(value))

    new_location.append(uri)
    new_location = 'http://smqtk:12345/' + '/'.join(new_location)

    log('PROXYING?')
    return proxy(new_location)


if __name__ == '__main__':
    filenames = next(os.walk('/data'))[2]
    for name in filenames:
        if name.startswith('.'):
            continue

        filepath = os.path.join('/data', name)
        if os.path.isfile(filepath):
            hasher = sha1()
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(1024)
                    if not chunk:
                        break
                    hasher.update(chunk)

            hasher = hasher.hexdigest()
            link_path = os.path.join('/links', hasher)
            if not os.path.exists(link_path):
                os.symlink('..' + filepath, link_path)

    app.run(host='0.0.0.0', port=12345, threaded=True)
