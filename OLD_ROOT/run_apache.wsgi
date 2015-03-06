"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import sys
import os

pathname, filename = os.path.split(os.path.abspath(__file__))
print pathname

import logging
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0, pathname)
sys.path.insert(1, os.path.join(pathname, 'Backend'))
from WebUI import app as application

