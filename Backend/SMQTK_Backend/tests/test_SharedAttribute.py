"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from SMQTK_Backend.ControllerProcess import ControllerProcess
from SMQTK_Backend.SharedAttribute import SharedAttribute, register_mdb_loc


class test_SharedAttribute (object):

    def setUp(self):
        register_mdb_loc('localhost', 27017)

    def test_bad_initialization(self):
        class bad_subclass (ControllerProcess):
            NAME = 'bad'
            foo = SharedAttribute()

            def _run(self):
                # does things here
                self.foo = 'bar'

        # noinspection PyTypeChecker
        t = bad_subclass('foo')
        t.start()
        t.join()
        # noinspection PyStatementEffect
        assert t.foo != 'bar' and t.foo is None

    def test_good_initialization(self):
        class good_subclass (ControllerProcess):
            NAME = 'bad'
            foo = SharedAttribute()

            def __init__(self):
                # noinspection PyTypeChecker
                super(good_subclass, self).__init__('foo')
                self.foo = 'bar'

            def _run(self):
                # does things here
                self.foo = 'bar'

        t = good_subclass()
        t.start()
        t.join()
        assert t.foo == 'bar'
