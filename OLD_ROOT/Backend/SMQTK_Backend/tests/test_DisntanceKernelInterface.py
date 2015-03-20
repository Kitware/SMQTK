"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
__author__ = 'paul.tunison'

from nose import tools
import os.path as osp
import unittest

from SMQTK_Backend.DistanceKernelInterface import (
    DistanceKernel_File_IQR,
    DistanceKernel_File_Archive
)


#noinspection PyPep8Naming
class test_DistanceKernel_IQR (unittest.TestCase):

    def setUp(self):
        data_dir = osp.abspath(osp.join(osp.dirname(__file__), 'data'))
        self.clipids_file = osp.join(data_dir, 'symmetric_clipids.txt')
        self.bgflags_file = osp.join(data_dir, 'symmetric_bgflags.txt')
        self.kernel_file = osp.join(data_dir, 'symmetric_distance_kernel.npy')

    def test_iqr_kernel_construction(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        assert iqr_kernel

    def test_submatrix_extraction_1(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        cid_map, bg_map, m = iqr_kernel.get_sub_matrix(50, 61, 97)

        self.assertEqual(len(cid_map), 43,
                         "(cid_map) Incorrect number of elements extracted "
                         "during symmetric sub-matrix extraction (%d != %d)"
                         % (len(cid_map), 43))
        self.assertEqual(len(bg_map), 43,
                         "(bg_map) Incorrect number of elements extracted "
                         "during symmetric sub-matrix extraction (%d != %d)"
                         % (len(cid_map), 43))
        self.assertEqual(m.shape, (43, 43),
                         "Incorrect matrix shape returned! (%s != %s"
                         % (m.shape, (43, 43)))
        self.assertEqual(bg_map.count(1), 40,
                         "Incorrect number of background labeled clips! "
                         "(%d != %d)"
                         % (bg_map.count(1), 40))

    def test_submatrix_extraction_2(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        cid_map, bg_map, m = iqr_kernel.get_sub_matrix(17, 39)

        self.assertEqual(len(cid_map), 40,
                         "(cid_map) Incorrect number of elements extracted "
                         "during symmetric sub-matrix extraction (%d != %d)"
                         % (len(cid_map), 40))
        self.assertEqual(len(bg_map), 40,
                         "(bg_map) Incorrect number of elements extracted "
                         "during symmetric sub-matrix extraction (%d != %d)"
                         % (len(cid_map), 40))
        self.assertEqual(m.shape, (40, 40),
                         "Incorrect matrix shape returned! (%s != %s"
                         % (m.shape, (40, 40)))
        self.assertEqual(bg_map.count(1), 38,
                         "Incorrect number of background labeled clips! "
                         "(%d != %d)"
                         % (bg_map.count(1), 38))

    def test_submatrix_extraction_3(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        cid_map, bg_map, m = iqr_kernel.get_sub_matrix()

        self.assertEqual(len(cid_map), 40,
                         "(cid_map) Incorrect number of elements extracted "
                         "during symmetric sub-matrix extraction (%d != %d)"
                         % (len(cid_map), 40))
        self.assertEqual(len(bg_map), 40,
                         "(bg_map) Incorrect number of elements extracted "
                         "during symmetric sub-matrix extraction (%d != %d)"
                         % (len(cid_map), 40))
        self.assertEqual(m.shape, (40, 40),
                         "Incorrect matrix shape returned! (%s != %s"
                         % (m.shape, (40, 40)))
        self.assertEqual(bg_map.count(1), 40,
                         "Incorrect number of background labeled clips! "
                         "(%d != %d)"
                         % (bg_map.count(1), 40))

    @tools.raises(AssertionError)
    def test_submatrix_extraction_fail_1(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        iqr_kernel.get_sub_matrix(46, 2756834)

    @tools.raises(AssertionError)
    def test_submatrix_extraction_fail_2(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        iqr_kernel.get_sub_matrix("not a number")

    def test_row_extraction_1(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        rowIDs, colIDs, m = iqr_kernel.extract_rows(2, 54, 90)

        self.assertEqual(len(rowIDs), 3)
        self.assertEqual(len(colIDs), len(iqr_kernel.edge_clip_ids[1]))
        self.assertEqual(m.shape, (3, len(iqr_kernel.edge_clip_ids[1])))

    def test_row_extraction_2(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        rowIDs, colIDs, m = iqr_kernel.extract_rows()
        # Numpy matrices can have a non-zero shape when they have no contents
        # like the (0, 100) shape here.

        self.assertEqual(len(rowIDs), 0,
                         "Incorrect number of rows returned (%d != %d)"
                         % (len(rowIDs), 0))
        self.assertEqual(len(colIDs), len(iqr_kernel.edge_clip_ids[1]),
                         "Incorrect number of columns returned (%d != %d)"
                         % (len(colIDs), len(iqr_kernel.edge_clip_ids[1])))
        self.assertEqual(m.shape, (0, len(iqr_kernel.edge_clip_ids[1])))

    @tools.raises(AssertionError)
    def test_row_extraction_failure_1(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        iqr_kernel.extract_rows(2, 23456)

    @tools.raises(AssertionError)
    def test_row_extraction_failure_2(self):
        iqr_kernel = DistanceKernel_File_IQR(self.clipids_file,
                                             self.bgflags_file,
                                             self.kernel_file)
        iqr_kernel.extract_rows(2, 'not a number')


#noinspection PyPep8Naming
class test_DistanceKernel_Archive (unittest.TestCase):

    def setUp(self):
        data_dir = osp.abspath(osp.join(osp.dirname(__file__), 'data'))
        self.rowIDs_file = osp.join(data_dir, 'asymmetric_clipids_rows.txt')
        self.colIDs_file = osp.join(data_dir, 'asymmetric_clipids_cols.txt')
        self.kernel_file = osp.join(data_dir, 'asymmetric_distance_kernel.npy')

    def test_arc_kernel_construction(self):
        arc_kernel = DistanceKernel_File_Archive(self.rowIDs_file,
                                                 self.colIDs_file,
                                                 self.kernel_file)
        assert arc_kernel

    @tools.raises(NotImplementedError)
    def test_submatrix_extraction(self):
        arc_kernel = DistanceKernel_File_Archive(self.rowIDs_file,
                                                 self.colIDs_file,
                                                 self.kernel_file)
        arc_kernel.get_sub_matrix(10, 50)

    def test_row_extraction_1(self):
        arc_kernel = DistanceKernel_File_Archive(self.rowIDs_file,
                                                 self.colIDs_file,
                                                 self.kernel_file)
        rowIDs, colIDs, m = arc_kernel.extract_rows()
        # Numpy matrices can have a non-zero shape when they have no contents
        # like the (0, 100) shape here.

        self.assertEqual(len(rowIDs), 0,
                         "Incorrect number of rows returned! (%d != %d)"
                         % (len(rowIDs), 0))
        self.assertEqual(len(colIDs), 10000,
                         "Incorrect number of columns returned! (%d != %d)"
                         % (len(colIDs), 10000))
        self.assertEqual(m.shape, (0, 10000),
                         "Incorrect matrix shape (%s, != %s)"
                         % (m.shape, (0, 10000)))

    def test_row_extraction_2(self):
        arc_kernel = DistanceKernel_File_Archive(self.rowIDs_file,
                                                 self.colIDs_file,
                                                 self.kernel_file)
        rowIDs, colIDs, m = arc_kernel.extract_rows(26, 73)
        # Numpy matrices can have a non-zero shape when they have no contents
        # like the (0, 100) shape here.

        self.assertEqual(len(rowIDs), 2,
                         "Incorrect number of rows returned! (%d != %d)"
                         % (len(rowIDs), 2))
        self.assertEqual(len(colIDs), 10000,
                         "Incorrect number of columns returned! (%d != %d)"
                         % (len(colIDs), 10000))
        self.assertEqual(m.shape, (2, 10000),
                         "Incorrect matrix shape (%s, != %s)"
                         % (m.shape, (2, 10000)))

    @tools.raises(AssertionError)
    def test_row_extraction_failure_1(self):
        arc_kernel = DistanceKernel_File_Archive(self.rowIDs_file,
                                                 self.colIDs_file,
                                                 self.kernel_file)
        arc_kernel.extract_rows(10083, 19968)

    @tools.raises(AssertionError)
    def test_row_extraction_failure_1(self):
        arc_kernel = DistanceKernel_File_Archive(self.rowIDs_file,
                                                 self.colIDs_file,
                                                 self.kernel_file)
        arc_kernel.extract_rows(10, "not a number")