from distutils.version import LooseVersion
import os
import pickle
import re
import unittest
import warnings

import numpy
import pytest
# move registered in ``smqtk.tests`` module __init__.
from six.moves import mock

from smqtk.algorithms.image_io.gdal_io import (
    cv2, osgeo,
    get_gdal_driver_supported_mimetypes,
    load_dataset_tempfile,
    load_dataset_vsimem,
    GdalImageReader,
    GdalRGBImageReader,
)
from smqtk.representation import DataElement
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR


IMG_PATH_GRACE_HOPPER = os.path.join(TEST_DATA_DIR, "grace_hopper.png")


@pytest.mark.skipif(osgeo is None,
                    reason="osgeo module not importable.")
class TestGdalHelperFunctions (unittest.TestCase):

    def test_gdal_supported_drivers(self):
        """
        Test that GDAL driver mimetype set return is non-zero.
        """
        ret = get_gdal_driver_supported_mimetypes()
        assert isinstance(ret, set)
        assert len(ret) > 0

    def test_gdal_supported_drivers_caching(self):
        """
        Test that GDAL driver mimetype getter performs caching.
        """
        # If the expected cache attr exists already on the function, remove it
        if hasattr(get_gdal_driver_supported_mimetypes, 'cache'):
            del get_gdal_driver_supported_mimetypes.cache
        assert not hasattr(get_gdal_driver_supported_mimetypes, 'cache')

        ret1 = get_gdal_driver_supported_mimetypes()

        # A second call to the function should return the same thing but not
        # call anything from GDAL.
        with mock.patch('smqtk.algorithms.image_io.gdal_io.gdal') as m_gdal:
            ret2 = get_gdal_driver_supported_mimetypes()
            assert ret2 == ret1
            m_gdal.GetDriverCount.assert_not_called()
            m_gdal.GetDriver.assert_not_called()

    def test_load_dataset_tempfile_actual(self):
        e = DataFileElement(IMG_PATH_GRACE_HOPPER, readonly=True)
        e.write_temp = mock.MagicMock(wraps=e.write_temp)
        e.clean_temp = mock.MagicMock(wraps=e.clean_temp)
        e.get_bytes = mock.MagicMock(wraps=e.get_bytes)

        # Using explicit patcher start/stop in order to avoid using ``patch``
        # as a decorator because ``osgeo`` might not be defined when
        # decorating the method.
        patcher_gdal_open = mock.patch('smqtk.algorithms.image_io.gdal_io.gdal'
                                       '.Open', wraps=osgeo.gdal.Open)
        self.addCleanup(patcher_gdal_open.stop)

        m_gdal_open = patcher_gdal_open.start()

        with load_dataset_tempfile(e) as gdal_ds:
            # noinspection PyUnresolvedReferences
            e.write_temp.assert_called_once_with()
            # noinspection PyUnresolvedReferences
            e.get_bytes.assert_not_called()

            m_gdal_open.assert_called_once()

            assert gdal_ds.RasterCount == 3
            assert gdal_ds.RasterXSize == 512
            assert gdal_ds.RasterYSize == 600

        # noinspection PyUnresolvedReferences
        e.clean_temp.assert_called_once_with()
        assert len(e._temp_filepath_stack) == 0

    def test_load_dataset_vsimem(self):
        """
        Test that VSIMEM loading context
        """
        if LooseVersion(osgeo.__version__).version[0] < 2:
            pytest.skip("Skipping VSIMEM test because GDAL version < 2",
                        # allow_module_level=True
                        )

        e = DataFileElement(IMG_PATH_GRACE_HOPPER, readonly=True)
        e.write_temp = mock.MagicMock(wraps=e.write_temp)
        e.clean_temp = mock.MagicMock(wraps=e.clean_temp)
        e.get_bytes = mock.MagicMock(wraps=e.get_bytes)

        vsimem_path_re = re.compile('^/vsimem/\w+$')

        # Using explicit patcher start/stop in order to avoid using ``patch``
        # as a decorator because ``osgeo`` might not be defined when
        # decorating the method.
        patcher_gdal_open = mock.patch(
            'smqtk.algorithms.image_io.gdal_io.gdal.Open',
            wraps=osgeo.gdal.Open,
        )
        self.addCleanup(patcher_gdal_open.stop)
        patcher_gdal_unlink = mock.patch(
            'smqtk.algorithms.image_io.gdal_io.gdal.Unlink',
            wraps=osgeo.gdal.Unlink,
        )
        self.addCleanup(patcher_gdal_unlink.stop)

        m_gdal_open = patcher_gdal_open.start()
        m_gdal_unlink = patcher_gdal_unlink.start()

        with load_dataset_vsimem(e) as gdal_ds:
            # noinspection PyUnresolvedReferences
            e.write_temp.assert_not_called()
            # noinspection PyUnresolvedReferences
            e.get_bytes.assert_called_once_with()

            m_gdal_open.assert_called_once()
            ds_path = gdal_ds.GetDescription()
            assert vsimem_path_re.match(ds_path)

            assert gdal_ds.RasterCount == 3
            assert gdal_ds.RasterXSize == 512
            assert gdal_ds.RasterYSize == 600

        m_gdal_unlink.assert_called_once_with(ds_path)
        # noinspection PyUnresolvedReferences
        e.clean_temp.assert_not_called()
        assert len(e._temp_filepath_stack) == 0


def test_GdalImageReader_usable():
    """
    Test that GdalImageReader class reports as usable when GDAL is importable.
    """
    # Mock module value of ``osgeo`` to something not None to simulate
    # something having been imported.
    with mock.patch.dict('smqtk.algorithms.image_io.gdal_io.__dict__',
                         {'osgeo': object()}):
        assert GdalImageReader.is_usable() is True


def test_GdalImageReader_not_usable_missing_osgeo():
    """
    Test that class reports as not usable when GDAL is not importable (set
    to None in module).
    """
    # Catch expected warnings to not pollute output.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        # Mock module value of ``osgeo`` to None.
        with mock.patch.dict('smqtk.algorithms.image_io.gdal_io.__dict__',
                             {'osgeo': None}):
            assert GdalImageReader.is_usable() is False


@pytest.mark.skipif(not GdalImageReader.is_usable(),
                    reason="GdalImageReader implementation is not usable in "
                           "the current environment.")
class TestGdalImageReader (unittest.TestCase):
    """
    Tests for ``GdalImageReader`` class.

    If this was not skipped, ``osgeo`` must have been importable.
    """

    def test_init_default(self):
        """
        Test that construction with default parameters works.
        """
        GdalImageReader()

    def test_init_bad_load_method(self):
        """
        Test that passing a load_method string that is not one of the
        expected values raises a ValueError.
        """
        with pytest.raises(ValueError, match="Given `load_method` not a valid "
                                             "value"):
            GdalImageReader(load_method="not a valid method")

    @mock.patch("smqtk.algorithms.image_io.gdal_io.osgeo")
    def test_init_vsimem_req_gdal_2_fail(self, m_osgeo):
        """
        Test that we get a RuntimeError when using load_method='vsimem' and
        the current GDAL wrapper version is < 2.
        """
        # Mock a GDAL version less than 2.
        m_osgeo.__version__ = "1.11.0"
        with pytest.raises(RuntimeError,
                           match="Load method '{}' was specified, "
                                 "but required GDAL version of 2 is not "
                                 "satisfied \(imported version: {}\)\."
                                 .format(GdalImageReader.LOAD_METHOD_VSIMEM,
                                         "1.11.0")):
            GdalImageReader(load_method=GdalImageReader.LOAD_METHOD_VSIMEM)

    @mock.patch("smqtk.algorithms.image_io.gdal_io.osgeo")
    def test_init_vsimem_req_gdal_2_pass(self, m_osgeo):
        """
        Test that we do NOT get an error when using load_method='vsimem'
        and GDAL reports a version greater than 2.
        """
        # Mock a GDAL version greater than 2.
        m_osgeo.__version__ = "2.4.0"
        GdalImageReader(load_method=GdalImageReader.LOAD_METHOD_VSIMEM)

    def test_serialization(self):
        """
        Test that we can serialize and deserialize the algorithm, maintaining
        configuration responses.
        """
        expected_load_method = GdalImageReader.LOAD_METHOD_TEMPFILE
        inst1 = GdalImageReader(load_method=expected_load_method)

        expected_config = {'load_method': expected_load_method}
        assert inst1.get_config() == expected_config

        buff = pickle.dumps(inst1)
        #: :type: GdalImageReader
        inst2 = pickle.loads(buff)
        assert inst2.get_config() == expected_config

    @mock.patch('smqtk.algorithms.image_io.gdal_io'
                '.get_gdal_driver_supported_mimetypes',
                wraps=get_gdal_driver_supported_mimetypes)
    def test_valid_content_types(self, m_ggdsm):
        """
        Test that valid_content_types refers to the helper method and
        returns the same thing.
        """
        expected_content_types = get_gdal_driver_supported_mimetypes()

        actual_content_types = GdalImageReader().valid_content_types()

        m_ggdsm.assert_called_once_with()
        assert actual_content_types == expected_content_types

    def test_load_as_matrix(self):
        """
        Test that load func calls the appropriate func in the
        context-managers dict and returns an appropriate ndarray.
        """
        mock_elem = mock.MagicMock(spec_set=DataElement)
        mock_loader = mock.MagicMock()

        reader = GdalImageReader()
        reader._load_method = 'test-method-name'

        with mock.patch.dict(reader.LOAD_METHOD_CONTEXTMANAGERS,
                             {'test-method-name': mock_loader}):
            ret = reader._load_as_matrix(mock_elem)

        mock_loader.assert_called_once_with(mock_elem)
        mock_loader().__enter__.assert_called_once()
        mock_loader().__enter__().ReadAsArray.assert_called_once()
        assert ret == mock_loader().__enter__().ReadAsArray()


def test_GdalRGBImageReader_usable():
    """
    Test that GdalRGBImageReader is usable when both GDAL and OpenCV are
    importable.
    """
    # Mock module values of ``osgeo`` and ``cv2`` to something not None to
    # simulate something having been imported.
    with mock.patch.dict('smqtk.algorithms.image_io.gdal_io.__dict__',
                         {'osgeo': object(), 'cv2': object()}):
        assert GdalRGBImageReader.is_usable() is True


def test_GdalRGBIMageReader_not_usable_missing_osgeo():
    """
    Test that GdalRGBImageReader is not usable when ``osgeo`` is not
    importable while OpenCV is.
    """
    # Catch expected warnings to not pollute output.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        # Mock module value of ``osgeo`` to None.
        with mock.patch.dict('smqtk.algorithms.image_io.gdal_io.__dict__',
                             {'osgeo': None, 'cv2': object()}):
            assert GdalRGBImageReader.is_usable() is False


def test_GdalRGBIMageReader_not_usable_missing_cv2():
    """
    Test that GdalRGBImageReader is not usable when ``cv2`` is not
    importable while GDAL is.
    """
    # Catch expected warnings to not pollute output.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        # Mock module value of ``osgeo`` to None.
        with mock.patch.dict('smqtk.algorithms.image_io.gdal_io.__dict__',
                             {'osgeo': object(), 'cv2': None}):
            assert GdalRGBImageReader.is_usable() is False


def test_GdalRGBIMageReader_not_usable_missing_osgeo_cv2():
    """
    Test that GdalRGBImageReader is not usable when ``osgeo`` AND ``cv2``
    are not importable.
    """
    # Catch expected warnings to not pollute output.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        # Mock module value of ``osgeo`` to None.
        with mock.patch.dict('smqtk.algorithms.image_io.gdal_io.__dict__',
                             {'osgeo': None, 'cv2': None}):
            assert GdalRGBImageReader.is_usable() is False


@pytest.mark.skipif(not GdalRGBImageReader.is_usable(),
                    reason="GdalRGBImageReader implementation is not usable "
                           "in the current environment.")
class TestGdalRGBImageReader (unittest.TestCase):
    """
    Tests for ``GdalRGBImageReader`` class.

    If this is not skipped, ``osgeo`` and ``cv2`` must have been importable.
    """

    @mock.patch('smqtk.algorithms.image_io.gdal_io.numpy.interp',
                wraps=numpy.interp)
    @mock.patch('smqtk.algorithms.image_io.gdal_io.GdalImageReader'
                '._load_as_matrix')
    def test_load_as_matrix_correct_transpose_uint8(self, m_gir_lam,
                                                    m_np_interp):
        """
        Test that a [channel, height, width] return from GdalImageReader
        is correctly transposed into [height, width, channel].

        No type casing should happen from uint8 to uint8
        """
        with mock.patch('smqtk.algorithms.image_io.gdal_io.cv2.cvtColor',
                        wraps=cv2.cvtColor) as m_cv2_cvtColor:
            # 200x100 3-channel test image.
            base_image_mat = numpy.ones((3, 100, 200), dtype=numpy.uint8)
            m_gir_lam.return_value = base_image_mat

            expected_mat = numpy.ones((100, 200, 3), dtype=numpy.uint8)

            m_elem = mock.MagicMock(spec_set=DataElement)
            reader = GdalRGBImageReader()
            ret_mat = reader._load_as_matrix(m_elem)

            m_cv2_cvtColor.assert_not_called()
            m_np_interp.assert_not_called()
            assert ret_mat.dtype == numpy.uint8
            numpy.testing.assert_allclose(ret_mat, expected_mat)

    @mock.patch('smqtk.algorithms.image_io.gdal_io.numpy.interp',
                wraps=numpy.interp)
    @mock.patch('smqtk.algorithms.image_io.gdal_io.GdalImageReader'
                '._load_as_matrix')
    def test_load_as_matrix_grayscale_convert_uint8(self, m_gir_lam,
                                                    m_np_interp):
        """
        Test that a grayscale-to-RGB conversion occurs when the native GDAL
        read is a grayscale matrix.
        """
        with mock.patch('smqtk.algorithms.image_io.gdal_io.cv2.cvtColor',
                        wraps=cv2.cvtColor) as m_cv2_cvtColor:
            # 200x100 single-channel test image.
            base_image_mat = numpy.ones((100, 200), dtype=numpy.uint8)
            m_gir_lam.return_value = base_image_mat

            expected_mat = numpy.ones((100, 200, 3), dtype=numpy.uint8)

            m_elem = mock.MagicMock(spec_set=DataElement)
            reader = GdalRGBImageReader()
            ret_mat = reader._load_as_matrix(m_elem)

            m_cv2_cvtColor.assert_called_once_with(base_image_mat,
                                                   cv2.COLOR_GRAY2RGB)
            m_np_interp.assert_not_called()
            assert ret_mat.dtype == numpy.uint8
            numpy.testing.assert_allclose(ret_mat, expected_mat)

    @mock.patch('smqtk.algorithms.image_io.gdal_io.numpy.interp',
                wraps=numpy.interp)
    @mock.patch('smqtk.algorithms.image_io.gdal_io.GdalImageReader'
                '._load_as_matrix')
    def test_load_as_matrix_too_many_channels(self, m_gir_lam, m_np_interp):
        """
        Test that a ValueError is raised if the base image matrix has more
        than 3 channels of data.
        """
        with mock.patch('smqtk.algorithms.image_io.gdal_io.cv2.cvtColor',
                        wraps=cv2.cvtColor) as m_cv2_cvtColor:
            # 100x200 16-channel test image.
            base_image_mat = numpy.ones((16, 100, 200), dtype=numpy.uint8)
            m_gir_lam.return_value = base_image_mat

            m_elem = mock.MagicMock(spec_set=DataElement)

            reader = GdalRGBImageReader()
            with pytest.raises(ValueError, match="Unexpected image channel "
                                                 "format \(expected 3, "
                                                 "found 16\)"):
                reader._load_as_matrix(m_elem)

            m_cv2_cvtColor.assert_not_called()
            m_np_interp.assert_not_called()

    @mock.patch('smqtk.algorithms.image_io.gdal_io.numpy.interp',
                wraps=numpy.interp)
    @mock.patch('smqtk.algorithms.image_io.gdal_io.GdalImageReader'
                '._load_as_matrix')
    def test_load_as_matrix_dtype_based_interpolation_uint16(self, m_gir_lam,
                                                             m_np_interp):
        """
        Test that when the base image matrix type is not uint8 (uint16 in
        this case), we interpolate the scale into the uint8 range.
        """
        with mock.patch('smqtk.algorithms.image_io.gdal_io.cv2.cvtColor',
                        wraps=cv2.cvtColor) as m_cv2_cvtColor:
            # 200x100 3-channel test image of uint16 range
            base_image_mat = numpy.ones((3, 100, 200), dtype=numpy.uint16)
            base_image_mat[:] = numpy.iinfo(numpy.uint16).max // 2
            m_gir_lam.return_value = base_image_mat

            expected_mat = numpy.ones((100, 200, 3), dtype=numpy.uint8)
            expected_mat[:] = numpy.iinfo(numpy.uint8).max // 2

            m_elem = mock.MagicMock(spec_set=DataElement)
            reader = GdalRGBImageReader()
            ret_mat = reader._load_as_matrix(m_elem)

            m_cv2_cvtColor.assert_not_called()
            m_np_interp.assert_called_once()
            assert ret_mat.dtype == numpy.uint8
            numpy.testing.assert_allclose(ret_mat, expected_mat)
