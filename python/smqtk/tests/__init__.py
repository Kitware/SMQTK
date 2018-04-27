import os

import six


# Centrally add the mock move
six.add_move(six.MovedModule('mock', 'mock', 'unittest.mock'))


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
