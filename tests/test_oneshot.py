"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'dhan'


import sys
import os
sys.path.append(os.path.abspath(".."))

from common_utils.sprite import create_sprite

def test_sprite_creation():
	# create_sprite(frame_path, width=200, num_frames=10):
    root_path = os.path.abspath(os.path.join(".","..","work","537a74e70a3ee13f9d8bf993"))
    create_sprite(root_path + "/images/456/123456")

if __name__ == "__main__":
	test_sprite_creation()
