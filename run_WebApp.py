#!/usr/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

#!/usr/bin/python
if __name__ == '__main__':
    from WebUI import app
    #app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)
    #app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=True)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
    #app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
