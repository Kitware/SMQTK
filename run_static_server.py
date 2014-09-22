#!/usr/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""


def main():
    import flask
    import mimetypes
    import os.path as osp

    import smqtk_config

    mimetypes.add_type('image/png', '.png')
    mimetypes.add_type('video/ogg', '.ogv')
    mimetypes.add_type('video/webm', '.webm')

    #print "[DEBUG] Setting static directory to:", smqtk_config.STATIC_DIR
    app = flask.Flask(__name__,
                      static_url_path='/static',
                      static_folder=smqtk_config.STATIC_DIR)

    @app.route('/static/data/clips/<path:filename>')
    def send_static_clip(filename):
        #print "[DEBUG] Request for filename:", filename
        #print "[DEBUG] calling send_from_directory:", (smqtk_config.STATIC_DIR, filename)
        return flask.send_from_directory(osp.join(smqtk_config.STATIC_DIR, 'data', 'clips'), filename)

    #app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=True)
    #app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=True)
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)
    #app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()

