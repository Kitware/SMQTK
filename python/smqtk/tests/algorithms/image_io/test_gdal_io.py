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
from smqtk.representation import AxisAlignedBoundingBox, DataElement
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.tests import TEST_DATA_DIR


GH_IMAGE_FP = None              # type: str
GH_FILE_ELEMENT = None          # type: DataFileElement
GH_CROPPED_IMAGE_FP = None      # type: str
GH_CROPPED_FILE_ELEMENT = None  # type: DataFileElement
GH_CROPPED_BBOX = None          # type: AxisAlignedBoundingBox


def setup_module():
    # Initialize test image paths/elements/associated crop boxes.
    global GH_IMAGE_FP, GH_FILE_ELEMENT, GH_CROPPED_IMAGE_FP, \
        GH_CROPPED_FILE_ELEMENT, GH_CROPPED_BBOX

    GH_IMAGE_FP = os.path.join(TEST_DATA_DIR, "grace_hopper.png")
    GH_FILE_ELEMENT = DataFileElement(GH_IMAGE_FP, readonly=True)
    assert GH_FILE_ELEMENT.content_type() == 'image/png'

    GH_CROPPED_IMAGE_FP = \
        os.path.join(TEST_DATA_DIR, 'grace_hopper.100x100+100+100.png')
    GH_CROPPED_FILE_ELEMENT = DataFileElement(GH_CROPPED_IMAGE_FP,
                                              readonly=True)
    assert GH_CROPPED_FILE_ELEMENT.content_type() == 'image/png'
    GH_CROPPED_BBOX = AxisAlignedBoundingBox([100, 100], [200, 200])


def teardown_module():
    # Clean any tempfiles from global elements of not already.
    GH_FILE_ELEMENT.clean_temp()
    GH_CROPPED_FILE_ELEMENT.clean_temp()


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

    def test_load_dataset_tempfile(self):
        """
        Test DataElement temporary file based context loader.
        """
        # Creating separate element from global so we can mock it up.
        e = DataFileElement(GH_IMAGE_FP, readonly=True)
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
            pytest.skip("Skipping VSIMEM test because GDAL version < 2")

        # Creating separate element from global so we can mock it up.
        e = DataFileElement(GH_IMAGE_FP, readonly=True)
        e.write_temp = mock.MagicMock(wraps=e.write_temp)
        e.clean_temp = mock.MagicMock(wraps=e.clean_temp)
        e.get_bytes = mock.MagicMock(wraps=e.get_bytes)

        vsimem_path_re = re.compile(r'^/vsimem/\w+$')

        # Using explicit patcher start/stop in order to avoid using ``patch``
        # as a *decorator* because ``osgeo`` might not be defined when
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
        with pytest.raises(ValueError, match=r"Load method provided not a "
                                             r"valid value \(given 'not a "
                                             r"valid method'\)\. Must be one "
                                             r"of: "):
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
                           match=r"Load method '{}' was specified, "
                                 r"but required GDAL version of 2 is not "
                                 r"satisfied \(imported version: {}\)\."
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

        Mocking (wrapping) `get_gdal_driver_supported_mimetypes` in order check
        that it is being called under the hood.
        """
        expected_content_types = get_gdal_driver_supported_mimetypes()

        actual_content_types = GdalImageReader().valid_content_types()

        m_ggdsm.assert_called_once_with()
        assert actual_content_types == expected_content_types

    def test_load_as_matrix_tempfile(self):
        """
        Test that whole image is loaded successfully using tempfile loader.
        """
        wrapped_temp_loader = mock.MagicMock(wraps=load_dataset_tempfile)
        wrapped_vsimem_loader = mock.MagicMock(wraps=load_dataset_vsimem)

        with mock.patch.dict(GdalImageReader.LOAD_METHOD_CONTEXTMANAGERS, {
                    GdalImageReader.LOAD_METHOD_TEMPFILE: wrapped_temp_loader,
                    GdalImageReader.LOAD_METHOD_VSIMEM: wrapped_vsimem_loader
                }):
            # Using tempfile load method
            reader = GdalImageReader(
                load_method=GdalImageReader.LOAD_METHOD_TEMPFILE
            )
            mat = reader._load_as_matrix(GH_FILE_ELEMENT)
            assert mat.shape == (3, 600, 512)

        wrapped_temp_loader.assert_called_once_with(GH_FILE_ELEMENT)
        wrapped_vsimem_loader.assert_not_called()

    def test_load_as_matrix_vsimem(self):
        """
        Test that whole image is loaded successfully using vsimem loader.
        """
        if LooseVersion(osgeo.__version__).version[0] < 2:
            pytest.skip("Skipping VSIMEM test because GDAL version < 2")

        wrapped_temp_loader = mock.MagicMock(wraps=load_dataset_tempfile)
        wrapped_vsimem_loader = mock.MagicMock(wraps=load_dataset_vsimem)

        with mock.patch.dict(GdalImageReader.LOAD_METHOD_CONTEXTMANAGERS, {
                    GdalImageReader.LOAD_METHOD_TEMPFILE: wrapped_temp_loader,
                    GdalImageReader.LOAD_METHOD_VSIMEM: wrapped_vsimem_loader
                }):
            # Using VSIMEM load method
            reader = GdalImageReader(
                load_method=GdalImageReader.LOAD_METHOD_VSIMEM
            )
            mat = reader._load_as_matrix(GH_FILE_ELEMENT)
            assert mat.shape == (3, 600, 512)

        wrapped_temp_loader.assert_not_called()
        wrapped_vsimem_loader.assert_called_once_with(GH_FILE_ELEMENT)

    def test_load_as_matrix_with_crop(self):
        """
        Test that the image is loaded with the correct crop region.

        We load two images: the original with a crop specified, and a
        pre-cropped image. The results of each load should be the same,
        indicating the correct region from the source image is extracted.
        """
        reader = GdalImageReader(
            load_method=GdalImageReader.LOAD_METHOD_TEMPFILE)
        cropped_actual = reader.load_as_matrix(GH_FILE_ELEMENT,
                                               pixel_crop=GH_CROPPED_BBOX)
        cropped_expected = reader.load_as_matrix(GH_CROPPED_FILE_ELEMENT)
        # noinspection PyTypeChecker
        numpy.testing.assert_allclose(cropped_actual, cropped_expected)

    def test_load_as_matrix_with_crop_not_in_bounds(self):
        """
        Test that error is raised when crop bbox is not fully within the image
        bounds.
        """
        inst = GdalImageReader()

        # Nowhere close
        bb = AxisAlignedBoundingBox([5000, 6000], [7000, 8000])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(GH_FILE_ELEMENT, pixel_crop=bb)

        # Outside left side
        bb = AxisAlignedBoundingBox([-1, 1], [2, 2])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(GH_FILE_ELEMENT, pixel_crop=bb)

        # Outside top side
        bb = AxisAlignedBoundingBox([1, -1], [2, 2])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(GH_FILE_ELEMENT, pixel_crop=bb)

        # Outside right side
        bb = AxisAlignedBoundingBox([400, 400], [513, 600])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(GH_FILE_ELEMENT, pixel_crop=bb)

        # Outside bottom side
        bb = AxisAlignedBoundingBox([400, 400], [512, 601])
        with pytest.raises(RuntimeError,
                           match=r"Crop provided not within input image\. "
                                 r"Image shape: \(512, 600\), crop: "):
            inst.load_as_matrix(GH_FILE_ELEMENT, pixel_crop=bb)


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


def test_GdalRGBImageReader_not_usable_missing_osgeo():
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


def test_GdalRGBImageReader_not_usable_missing_cv2():
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


def test_GdalRGBImageReader_not_usable_missing_osgeo_cv2():
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
    def test_load_as_matrix_grayscale_convert_uint8(self, m_gir_lam,
                                                    m_np_interp):
        """
        Test that a grayscale-to-RGB conversion occurs when the native GDAL
        read is a grayscale matrix.
        """
        # Must patch cv2 function within test because ``cv2`` may not be defined
        # in outer scope as it is an optional dep
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
    def test_load_as_matrix_correct_transpose_uint8(self, m_gir_lam,
                                                    m_np_interp):
        """
        Test that a [channel, height, width] return from GdalImageReader
        is correctly transposed into [height, width, channel].

        No type casing should happen from uint8 to uint8
        """
        # Must patch cv2 function within test because ``cv2`` may not be defined
        # in outer scope as it is an optional dep
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

    @mock.patch('smqtk.algorithms.image_io.gdal_io.GdalImageReader'
                '._load_as_matrix')
    def test_load_as_matrix_incorrect_dims(self, m_gir_lam):
        """
        Test that a ValueError is raised if the parent ``_load_as_matrix``
        method returns a matrix with 1 or more than 4 dimensions (matrix
        dimensions, not channels).
        """
        m_elem = mock.MagicMock(spec_set=DataElement)
        reader = GdalRGBImageReader()

        # 1-dim matrix (simple vector) return.
        base_image_mat = numpy.ones(8, dtype=numpy.uint8)
        m_gir_lam.return_value = base_image_mat
        with pytest.raises(ValueError, match=r"Image matrix should have "
                                             r"dimensionality \[height, width\]"
                                             r" or \[height, width, channel\] "
                                             r"\(ndim = 2 or 3\), but instead "
                                             r"found ndim = 1\."):
            reader._load_as_matrix(m_elem)

        # 4-dim matrix return.
        base_image_mat = numpy.ones((1, 2, 3, 4), dtype=numpy.uint8)
        m_gir_lam.return_value = base_image_mat
        with pytest.raises(ValueError, match=r"Image matrix should have "
                                             r"dimensionality \[height, width\]"
                                             r" or \[height, width, channel\] "
                                             r"\(ndim = 2 or 3\), but instead "
                                             r"found ndim = 4\."):
            reader._load_as_matrix(m_elem)

    @mock.patch('smqtk.algorithms.image_io.gdal_io.numpy.interp',
                wraps=numpy.interp)
    @mock.patch('smqtk.algorithms.image_io.gdal_io.GdalImageReader'
                '._load_as_matrix')
    def test_load_as_matrix_too_many_channels(self, m_gir_lam, m_np_interp):
        """
        Test that a ValueError is raised if the base image matrix has more
        than 3 channels of data.
        """
        # Must patch cv2 function within test because ``cv2`` may not be defined
        # in outer scope as it is an optional dep
        with mock.patch('smqtk.algorithms.image_io.gdal_io.cv2.cvtColor',
                        wraps=cv2.cvtColor) as m_cv2_cvtColor:
            # 100x200 16-channel test image.
            base_image_mat = numpy.ones((16, 100, 200), dtype=numpy.uint8)
            m_gir_lam.return_value = base_image_mat

            m_elem = mock.MagicMock(spec_set=DataElement)

            reader = GdalRGBImageReader()
            with pytest.raises(ValueError, match=r"Unexpected image channel "
                                                 r"format \(expected 3, "
                                                 r"found 16\)"):
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
        # Must patch cv2 function within test because ``cv2`` may not be defined
        # in outer scope as it is an optional dep
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
