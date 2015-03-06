"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

# Making sure that the environment has been setup
import os
if not os.environ.get('SMQTK_BACKEND_SETUP'):
    raise RuntimeError("SMQTK Environment not setup!")
del os
