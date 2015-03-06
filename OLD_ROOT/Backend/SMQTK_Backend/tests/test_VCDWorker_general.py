"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'paul.tunison'


import logging
import nose.tools as tools
import numpy
import os
import os.path as osp

from SMQTK_Backend.VCDStore.VCDStoreElement import VCDStoreElement
from SMQTK_Backend.VCDWorkers import VCDWorkerInterface
from SMQTK_Backend.VCDWorkers.descriptor_modules import get_descriptor_modules


#noinspection PyMethodMayBeStatic
class test_VCDWorker_general (object):

    #noinspection PyAttributeOutsideInit
    def setUp(self):
        self.log = logging.getLogger('test_VCDWorker_general')
        self.work_dir = osp.join(osp.dirname(__file__), 'work')

    def test_plugin_importer(self):
        """
        testing plugin gather method, which tests that all plugins are
        structured correctly that retrieval of a known dummy class works.
        """
        descr_modules = get_descriptor_modules()
        tools.assert_in('Dummy', descr_modules.keys())
        tools.assert_true(issubclass(descr_modules['Dummy'],
                                     VCDWorkerInterface))

    def test_dummy(self):
        """
        Test creating dummy instance and methods, including interface methods.
        Default configuration parameters should be functional.
        """
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        elem = dummy_inst.process_video('/foo/bar/HVC123456.mp4')

        self.log.info("VCDStoreElement type: %s", VCDStoreElement)
        tools.assert_is_instance(elem, VCDStoreElement)
        tools.assert_equal(elem.descriptor_id, "Dummy")
        tools.assert_equal(elem.video_id, 123456)
        #noinspection PyTypeChecker
        # reason -> because this is how numpy works
        tools.assert_true(all(elem.feat_vec == numpy.array((1, 2, 3, 4, 5))))

    @tools.raises(AttributeError)
    def test_interface_get_prefix_fail(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        # Expects an HVC formatted file name
        dummy_inst.get_video_prefix("foobat123456.mp4")

    def test_interface_get_prefix_rel(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        # Relative path
        pfx, key = dummy_inst.get_video_prefix('HVC764768.mp4')
        tools.assert_equal(pfx, '768')
        tools.assert_equal(key, '764768')

    def test_interface_get_prefix_abs(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        # Abs path
        pfx, key = dummy_inst.get_video_prefix('/home/user/data/HVC904576.flv')
        tools.assert_equal(pfx, '576')
        tools.assert_equal(key, '904576')

    def test_interface_get_prefix_short(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        # not 6-digit key
        pfx, key = dummy_inst.get_video_prefix('HVC5761.video')
        tools.assert_equal(pfx, '761')
        tools.assert_equal(key, '5761')

    def test_interface_tempify_filename(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        t = dummy_inst.tempify_filename('foo.txt')
        tools.assert_equal(t, 'foo.txt.TEMP')

    def test_interface_create_dir(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        dirpath = osp.join(osp.dirname(__file__), 'work/example_dir')
        tools.assert_false(osp.isdir(dirpath))
        p = dummy_inst.create_dir(dirpath)
        tools.assert_true(osp.isabs(p))
        tools.assert_true(osp.isdir(dirpath))
        tools.assert_true(osp.isdir(p))
        os.rmdir(dirpath)  # clean-up

    def test_interface_create_dir_existing(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        dirpath = osp.join(osp.dirname(__file__), 'work')
        tools.assert_true(osp.isdir(dirpath))
        p = dummy_inst.create_dir(dirpath)
        tools.assert_true(osp.isabs(p))
        tools.assert_true(osp.isdir(dirpath))
        tools.assert_true(osp.isdir(p))

    @tools.raises(OSError)
    def test_interface_create_dir_fail(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        dirpath = osp.join(osp.dirname(__file__), 'work/existing')
        tools.assert_true(osp.isfile(dirpath))
        dummy_inst.create_dir(dirpath)

    #noinspection PyProtectedMember
    def test_frame_predictor(self):
        dummy_class = get_descriptor_modules()['Dummy']
        dummy_inst = dummy_class(dummy_class.generate_config(), self.work_dir)

        frames = dummy_inst._get_frames_for_interval(10, 1, 0, 1.0)
        tools.assert_equal(frames, range(10))

        frames = dummy_inst._get_frames_for_interval(10, 1, 0, 1.0, 3)
        tools.assert_equal(frames, range(3))

        frames = dummy_inst._get_frames_for_interval(600, 20, 1.5, 1.0,  10.0)
        tools.assert_equal(frames, [30, 50, 70, 90, 110, 130, 150, 170, 190])
