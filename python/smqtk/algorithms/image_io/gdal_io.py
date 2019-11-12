import collections
from contextlib import contextmanager
from distutils.version import LooseVersion
import tempfile
import warnings

import numpy as np
import six
from six.moves import range

from smqtk.algorithms import ImageReader
from smqtk.utils.image import crop_in_bounds

try:
    import osgeo
    import osgeo.gdal as gdal
    import osgeo.gdal_array as gdal_array
except ImportError:
    osgeo = gdal = gdal_array = None


###############################################################################
# Plugin helper function(s)
#
# TODO: Move appropriate functions to a ``smqtk.utils.gdal`` module.

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


def get_possible_gdal_gci_values():
    """
    Get the set of possible gdal.GCI_* values.

    This function caches the constructed set as this should not change during
    the runtime of an application.

    :return: The set of possible gdal.GCI_* values.
    :rtype: set[int]
    """
    try:
        return get_possible_gdal_gci_values.cache
    except AttributeError:
        get_possible_gdal_gci_values.cache = s = \
            set(map(lambda a: getattr(gdal, a),
                    [attr for attr in dir(gdal) if attr.startswith("GCI_")]))
        return s


def get_gdal_gci_abbreviation_map():
    """
    Abbreviation mapping::

        'a' = gdal.GCI_AlphaBand
        'b' = gdal.GCI_BlueBand
        'g' = gdal.GCI_GreenBand
        'r' = gdal.GCI_RedBand
        'c' = gdal.GCI_CyanBand
        'm' = gdal.GCI_MagentaBand
        'y' = gdal.GCI_YellowBand
        'h' = gdal.GCI_HueBand
        's' = gdal.GCI_SaturationBand
        'l' = gdal.GCI_LightnessBand

    This function caches the constructed set as this should not change during
    the runtime of an application.

    :return: the new or cached character-to-GCI value map for channels for which
        a single letter abbreviation makes sense.  Character keys are in lower
        case.
    :rtype: dict[str, int]
    """
    try:
        return get_gdal_gci_abbreviation_map.map_cache
    except AttributeError:
        get_gdal_gci_abbreviation_map.map_cache = m = {
            'a': gdal.GCI_AlphaBand,
            'b': gdal.GCI_BlueBand,
            'g': gdal.GCI_GreenBand,
            'r': gdal.GCI_RedBand,
            'c': gdal.GCI_CyanBand,
            'm': gdal.GCI_MagentaBand,
            'y': gdal.GCI_YellowBand,
            'h': gdal.GCI_HueBand,
            's': gdal.GCI_SaturationBand,
            'l': gdal.GCI_LightnessBand
        }
        return m


def map_gci_list_to_names(gci_list):
    """
    Translate a sequence of GDAL GCI values into a list of their string names.

    Pre-condition: Integers provided are valid color interpretation constants.

    :param collections.Iterable[int] gci_list:
        Integer GDAL color interpretation integer constants sequence.
    :return: List of o the string names for each color interpretation constant
        provided
    :rtype: list[str]
    """
    return [gdal.GetColorInterpretationName(gci_int) for gci_int in gci_list]


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
    # Unguarded next() call is OK in this case because the generator returned
    # by ``_get_candidate_names()`` does not terminate.
    # noinspection PyProtectedMember
    tmp_vsimem_path = '/vsimem/{}'.format(
        six.next(tempfile._get_candidate_names())  # lgtm[py/unguarded-next-in-generator]
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
    the [height, width, channel] format for multi-channel imagery and just
    [height, width] for single channel imagery.
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

    def __init__(self, load_method=LOAD_METHOD_TEMPFILE, channel_order=None):
        """
        Use GDAL to read raster image pixel data and returns an image matrix in
        the format native to the input data.

        Channel Order
        -------------
        A custom selection of input image channels as well as their order may
        be specified by the ``channel_order`` parameter.  This may either be a
        string that uses channel abbreviations or a sequence of integers
        specifying GDAL GCI constants (``osgeo.gdal.GCI_*``).  If a string is
        provided, we cast to lower case as a standardization.

        If the source image does not report color interpretations for its bands
        then any specification of ``channel_order`` will result in an error when
        ``load_as_matrix`` is called (unable to map requested channels to bands
        reporting as type 0, or "unknown").

        See :func:`get_gdal_gci_abbreviation_map` for supported abbreviations.

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
        :param str | collections.Sequence[int] channel_order:
            Optional specific selection and order of channels read from the
            image to be included in the output matrix.  See above for more
            details.

        :raises RuntimeError: The ``vsimem`` load method was specified but the
            currently imported GDAL version is not >= 2.

        """
        super(GdalImageReader, self).__init__()

        self._channel_order = channel_order
        # The channel order in known GDAL GCI integer values. This is None when
        # channel_order is none, otherwise is a sequence of valid GDAL GCI
        # integer values.
        #: :type: None | list[int]
        self._channel_order_gci = None
        if channel_order is not None:
            # Is Sequence check.
            if not isinstance(channel_order, collections.Sequence):
                raise ValueError("Channel order must be a sequence in order to "
                                 "discern order! Given type: {}"
                                 .format(type(channel_order)))
            # Cannot be given an empty order sequence.
            if len(channel_order) == 0:
                raise ValueError("Invalid channel order, must request at least "
                                 "one band. Given: '{}'".format(channel_order))
            # If using an abbreviation string, make sure all characters match a
            # known abbreviation.
            if isinstance(channel_order, six.string_types):
                self._channel_order = channel_order = channel_order.lower()
                abb_map = get_gdal_gci_abbreviation_map()
                # Set will be non-empty if there is an invalid character.
                valid_set = set(abb_map)
                # noinspection PyTypeChecker
                diff_set = \
                    set(channel_order).difference(valid_set)
                if diff_set:
                    raise ValueError("Invalid abbreviation character in given "
                                     "channel order. Invalid characters: {}. "
                                     "Valid characters: {}."
                                     .format(list(diff_set), list(valid_set)))
                # Cache channel order in GCI translated values
                self._channel_order_gci = [abb_map[a] for a in channel_order]
            # When not string, make sure integer values are in the reported set
            # values from GDAL, otherwise we'll get a runtime error later.
            else:
                valid_set = set(get_possible_gdal_gci_values())
                diff_set = set(channel_order).difference(valid_set)
                if diff_set:
                    raise ValueError("Invalid GDAL band integer values given. "
                                     "Given invalid values: {}."
                                     .format(list(diff_set)))
                # Given channel order was already in GCI values, mirror in
                # expected attribute.
                self._channel_order_gci = list(channel_order)

        self._load_method = load_method.lower()
        # 1. Check that the given load method is one that we support.
        # 2. Check that GDAL version >= 2 if VSIMEM method specified
        if self._load_method not in self.LOAD_METHOD_CONTEXTMANAGERS:
            raise ValueError("Load method provided not a valid value (given "
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
            'channel_order': self._channel_order,
        }

    def valid_content_types(self):
        """
        :return: A set valid MIME types that are "valid" within the implementing
            class' context.
        :rtype: set[str]
        """
        return get_gdal_driver_supported_mimetypes()

    def _load_as_matrix(self, data_element, pixel_crop=None):
        """
        Internal method to be implemented that attempts loading an image
        from the given data element and returning it as an image matrix.

        Pre-conditions:
            - ``pixel_crop`` has a non-zero volume and is composed of integer
              types.

        :param smqtk.representation.DataElement data_element:
            DataElement to load image data from.
        :param None|smqtk.representation.AxisAlignedBoundingBox pixel_crop:
            Optional pixel crop region to load from the given data.  If this
            is provided it must represent a valid sub-region within the loaded
            image, otherwise a RuntimeError is raised.

        :raises RuntimeError: A crop region was specified but did not specify a
            valid sub-region of the image.

        :return: Numpy ndarray of the image data. Specific return format is
            implementation dependant.
        :rtype: np.ndarray

        """
        if data_element.is_empty():
            raise ValueError("{} cannot load 0-sized data (no bytes in {})."
                             .format(self.name, data_element))
        load_cm = self.LOAD_METHOD_CONTEXTMANAGERS[self._load_method]
        with load_cm(data_element) as gdal_ds:  # type: gdal.Dataset
            img_width = gdal_ds.RasterXSize
            img_height = gdal_ds.RasterYSize

            # GDAL wants [x, y, width, height] as the first 4 positional
            # arguments to ``ReadAsArray``.
            xywh = [0, 0, img_width, img_height]
            if pixel_crop:
                if not crop_in_bounds(pixel_crop, img_width, img_height):
                    raise RuntimeError("Crop provided not within input image. "
                                       "Image shape: {}, crop: {}"
                                       .format((img_width, img_height),
                                               pixel_crop))
                # This is testing faster than ``np.concatenate``.
                xywh = \
                    pixel_crop.min_vertex.tolist() + pixel_crop.deltas.tolist()

            # Select specific channels if they are present in this dataset, or
            # just get all of them
            if self._channel_order is not None:
                # Map raster bands from CI value to band index.
                # - GDAL uses 1-based indexing.
                band_ci_to_idx = {
                    gdal_ds.GetRasterBand(b_i).GetColorInterpretation(): b_i
                    for b_i in range(1, gdal_ds.RasterCount+1)
                }
                gci_diff = \
                    set(self._channel_order_gci).difference(band_ci_to_idx)
                if gci_diff:
                    raise RuntimeError(
                        "Data element did not provide channels required to "
                        "satisfy requested channel order {}.  "
                        "Data had bands: {} (missing {})."
                        .format(map_gci_list_to_names(self._channel_order_gci),
                                map_gci_list_to_names(band_ci_to_idx),
                                map_gci_list_to_names(gci_diff)))
                # Initialize a matrix to read band image data into
                # TODO: Handle when there are no bands?
                band_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
                    gdal_ds.GetRasterBand(1).DataType
                )
                if len(self._channel_order_gci) > 1:
                    img_mat = np.ndarray([xywh[3], xywh[2],
                                          len(self._channel_order_gci)],
                                         dtype=band_dtype)
                    for i, gci in enumerate(self._channel_order_gci):
                        #: :type: gdal.Band
                        b = gdal_ds.GetRasterBand(band_ci_to_idx[gci])
                        b.ReadAsArray(*xywh, buf_obj=img_mat[:, :, i])
                else:
                    img_mat = np.ndarray([xywh[3], xywh[2]],
                                         dtype=band_dtype)
                    gci = self._channel_order_gci[0]
                    b = gdal_ds.GetRasterBand(band_ci_to_idx[gci])
                    b.ReadAsArray(*xywh, buf_obj=img_mat)
            else:
                img_mat = gdal_ds.ReadAsArray(*xywh)
                if img_mat.ndim > 2:
                    # Transpose into [height, width, channel] format.
                    img_mat = img_mat.transpose(1, 2, 0)

        return img_mat
