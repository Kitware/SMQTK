from contextlib import contextmanager
from distutils.version import LooseVersion
import tempfile
import warnings

import numpy
import six
from six.moves import range

from smqtk.algorithms import ImageReader

try:
    import osgeo
    import osgeo.gdal as gdal
except ImportError:
    osgeo = gdal = None

try:
    import cv2
except ImportError:
    cv2 = None


###############################################################################
# Plugin helper function(s)

def get_gdal_driver_supported_mimetypes():
    """
    Get a set of mimetype strings that currently available GDAL drivers
    support.

    :return: Set of MIMETYPE string.
    :rtype: set[str]

    """
    # look for DMD_MIMETYPE metadata key in available drivers (available in
    # both versions 1 and 2)
    try:
        return get_gdal_driver_supported_mimetypes.cache
    except AttributeError:
        m_set = get_gdal_driver_supported_mimetypes.cache = set()
        m_key = gdal.DMD_MIMETYPE

        for i in range(gdal.GetDriverCount()):
            d = gdal.GetDriver(i)
            # A driver might be for a file type that has no associated
            # mimetype.
            d_mimetype = d.GetMetadata().get(m_key, None)
            if d_mimetype is not None:
                m_set.add(d_mimetype)

        return get_gdal_driver_supported_mimetypes.cache


###############################################################################
# Dataset load context managers
#
# These functions should be decorated with ``@contextmanager``, take in a
# single positional argument that is a SMQTK DataElement instance, and yield a
# GDAL Dataset instance. Functions should clean-up after themselves in their
# ``finally`` clause.
#

@contextmanager
def load_dataset_tempfile(data_element):
    """
    Load GDAL Dataset from element by first writing it to a temporary file.

    :param smqtk.representation.DataElement data_element:
        Element to load dataset from.

    :return: GDAL Dataset
    :rtype: gdal.Dataset

    """
    fp = data_element.write_temp()
    try:
        yield gdal.Open(fp)
    finally:
        data_element.clean_temp()


@contextmanager
def load_dataset_vsimem(data_element):
    """
    Load GDAL dataset from element by writing its bytes to a virtual file
    and loading a dataset from that virtual file.

    Requires GDAL major version 2 or greater.

    :param smqtk.representation.DataElement data_element:
        Element to load dataset from.

    :return: GDAL Dataset
    :rtype: gdal.Dataset

    """
    # noinspection PyProtectedMember
    tmp_vsimem_path = '/vsimem/{}'.format(
        six.next(tempfile._get_candidate_names())
    )
    gdal.FileFromMemBuffer(tmp_vsimem_path, data_element.get_bytes())
    try:
        yield gdal.Open(tmp_vsimem_path)
    finally:
        rc = gdal.Unlink(tmp_vsimem_path)
        if rc != 0:
            raise RuntimeError("Failed to gdal.Unlink virtual file '{}' "
                               "containing bytes from {}."
                               .format(tmp_vsimem_path, data_element))


###############################################################################
# Base GDAL reader classes


class GdalImageReader (ImageReader):
    """
    Use GDAL to read raster image pixel data and returns an image matrix in
    the format native to the input data.
    """

    LOAD_METHOD_TEMPFILE = 'tempfile'
    LOAD_METHOD_VSIMEM = 'vsimem'
    LOAD_METHOD_CONTEXTMANAGERS = {
        LOAD_METHOD_TEMPFILE: load_dataset_tempfile,
        LOAD_METHOD_VSIMEM: load_dataset_vsimem,
    }

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional
        dependencies that may not yet be available on the system,
        this method should check for those dependencies and return a
        boolean saying if the implementation is usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should
              emit a warning detailing why the implementation is not
              available for use.

        :return: Boolean determination of whether this implementation is
            usable.
        :rtype: bool

        """
        if osgeo is None:
            warnings.warn("GdalImageReader implementation not available "
                          "because ``osgeo`` is not importable.")
            return False
        return True

    def __init__(self, load_method=LOAD_METHOD_VSIMEM):
        """
        Use GDAL to read raster image pixel data and returns an image matrix in
        the format native to the input data.

        Load methods
        ------------

        ``tempfile``
        ^^^^^^^^^^^^
        Loader that writes the DataElement's bytes to a temporary file on
        disk first and then "opens" that file as a GDAL Dataset. This
        method performs double the I/O operations (write to disk and
        then immediately read from it again) but uses less RAM (only
        one "copy" of the image is loaded at a time) and is universally
        supported across GDAL versions.

        ``vsimem``
        ^^^^^^^^^^
        This method is more efficient in regards to run-time, however two
        copies of the image are loaded into RAM (the bytes of image for the
        virtual file and then image matrix itself). This method is also only
        functional in GDAL version 2 and above currently. A RuntimeError is
        raised if the currently imported GDAL is not version 2 or greater.

        :param str load_method:
            Method of loading GDAL Dataset from a DataElement.  This must be
            one of the ``GdalImageReader.LOAD_METHOD_*`` options.

        :raises RuntimeError: The ``vsimem`` load method was specified but the
            currently imported GDAL version is not >= 2.

        """
        super(GdalImageReader, self).__init__()

        self._load_method = load_method.lower()
        # 1. Check that the given load method is one that we support.
        # 2. Check that GDAL version >= 2 if VSIMEM method specified
        if self._load_method not in self.LOAD_METHOD_CONTEXTMANAGERS:
            raise ValueError("Given `load_method` not a valid value (given "
                             "'{}'). Must be one of: {}."
                             .format(load_method,
                                     self.LOAD_METHOD_CONTEXTMANAGERS))
        elif self._load_method == self.LOAD_METHOD_VSIMEM:
            gdal_major_version = LooseVersion(osgeo.__version__).version[0]
            if gdal_major_version < 2:
                raise RuntimeError("Load method '{}' was specified, "
                                   "but required GDAL version of 2 is not "
                                   "satisfied (imported version: {})."
                                   .format(self._load_method,
                                           osgeo.__version__))

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        :rtype: dict

        """
        return {
            'load_method': self._load_method,
        }

    def valid_content_types(self):
        """
        :return: A set valid MIME types that are "valid" within the implementing
            class' context.
        :rtype: set[str]
        """
        return get_gdal_driver_supported_mimetypes()

    def _load_as_matrix(self, data_element):
        """
        Internal method to be implemented that attempts loading an image
        from the given data element and returning it as an image matrix.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: numpy.ndarray

        """
        load_cm = self.LOAD_METHOD_CONTEXTMANAGERS[self._load_method]
        with load_cm(data_element) as gdal_ds:
            img_mat = gdal_ds.ReadAsArray()
        # Simply return matrix from basic GDAL read
        return img_mat


class GdalRGBImageReader(GdalImageReader):
    """
    Extension of ``GdalNativeImageReader`` to convert native image data
    into a uint8-type RGB format, with dimension format [height, width,
    channel]. This implementation additionally requires OpenCV python
    bindings for image matrix conversion.
    """

    @classmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.

        Since certain plugin implementations may require additional
        dependencies that may not yet be available on the system,
        this method should check for those dependencies and return a
        boolean saying if the implementation is usable.

        NOTES:
            - This should be a class method
            - When an implementation is deemed not usable, this should
              emit a warning detailing why the implementation is not
              available for use.

        :return: Boolean determination of whether this implementation is
            usable.
        :rtype: bool

        """
        usable = True
        # Required both gdal AND OpenCV
        if not super(GdalRGBImageReader, cls).is_usable():
            warnings.warn("GdalRGBImageReader implementation not available "
                          "because ``GdalImageReader`` does not report as "
                          "usable .")
            usable = False
        if cv2 is None:
            warnings.warn("GdalRGBImageReader implementation not available "
                          "because ``cv2`` is not importable.")
            usable = False
        return usable

    def _load_as_matrix(self, data_element):
        """
        Internal method to be implemented that attempts loading an image
        from the given data element and returning it as an image matrix.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.

        :raises ValueError: DataElement input represents an image with
            neither 2 or 3 dimensions, or more than 3 channels (not handily
            convertible to RGB).

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: numpy.ndarray

        """
        # Return raster image matrix in a 3-channel RGB representation.

        # Load raw raster matrix via GDAL
        mat = super(GdalRGBImageReader, self)._load_as_matrix(data_element)

        # if shape is only size 2, we have a monochrome image; convert to RGB
        if mat.ndim == 2 or (mat.ndim == 3 and mat.shape[2] == 1):
            #: :type: numpy.ndarray
            mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2RGB)
        elif mat.ndim == 3:
            # GDAL loads image arrays in [channel, height, width] format and we
            # want to return [height, width, channel].
            # noinspection PyTypeChecker
            # - this is actually an adequate method of calling `transpose`
            mat = mat.transpose(1, 2, 0)
        else:
            raise ValueError("Image matrix should have dimensionality "
                             "[height, width, channel] (ndim = 3), "
                             "but instead found ndim = {}."
                             .format(mat.ndim))

        if mat.shape[2] != 3:
            raise ValueError("Unexpected image channel format (expected 3, "
                             "found {})".format(mat.shape[2]))

        if mat.dtype != numpy.uint8:
            self._log.info("Rescaling input image from source type {} to "
                           "uint8 type.".format(mat.dtype))
            # NOTE: Instead of requiring integer type, Could check for
            #       isinstance of numpy.core.inexact (float) and then assume
            #       range of [0,1]? (not sure of other float image ranges)
            src_info = numpy.iinfo(mat.dtype)
            byte_info = numpy.iinfo(numpy.uint8)
            mat = numpy.interp(mat, [src_info.min, src_info.max],
                               [byte_info.min, byte_info.max])
            numpy.round(mat, out=mat)  # in-place
            mat = mat.astype(numpy.uint8)

        return mat
