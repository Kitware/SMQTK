"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from nose.tools import raises
import time

from SMQTK_Backend.ControllerProcess import ControllerProcess


class test_ControllerProcess (object):

    def setUp(self):
        class test_subclass (ControllerProcess):
            def _run(self):
                time.sleep(0.1)

        #noinspection PyAttributeOutsideInit
        self.test_proc_class = test_subclass

    #noinspection PyMethodMayBeStatic
    @raises(TypeError)
    def test_abs_invalid_construction(self):
        # This should throw an exception because it is an abstract base class
        ControllerProcess('foo')

    def test_uuids(self):
        n = 1000
        uuid_set = set()
        for i in xrange(n):
            t = self.test_proc_class(str(i))
            uuid_set.add(t.uuid)
        assert len(uuid_set) == n, "A duplicate UUID was generated"

    def test_proc_info(self):
        name = 'This Is a Name'
        t = self.test_proc_class(name)
        not_on_info = t.get_info()
        assert not_on_info.name == name
        assert not_on_info.uuid == t.uuid
        assert not not_on_info.is_alive
        t.start()
        on_info = t.get_info()
        assert on_info.name == name
        assert on_info.uuid == t.uuid
        assert on_info.is_alive
        t.join()
