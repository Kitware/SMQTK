"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


See the associated README.rst file in this directory for background.

"""

from .VCDStoreElement import VCDStoreElement
from .implementations import VCDStoreBackendInterface, SQLiteVCDStoreBackend


class VCDStore (object):
    """
    Abstract storage of feature vector information produced by descriptors.

    Contents are stored in VCDStoreElement objects which encapsulate a
    feature vector and its associated metadata (i.e. descriptor, video ID, etc.)

    No valid metadata value should be negative, so to denote 'no value' for a
    particular metadata field, a None value is used.

    """

    def __init__(self, _backend=SQLiteVCDStoreBackend,
                 *args, **kwargs):
        """
        Abstract storage of feature vector information produced by descriptors.

        Contents are stored in a key-value schema, where the key is the
        descriptor ID in combination with one or more of the video ID, frame
        number, and spacial pixel bounding region.

        By default, an SQLite3 backend is used, but a different one may be
        specified through the ``_backend`` keyword argument, which should be
        passed the class of the backend to use (from the implementations
        directory in this module). The SQLite3 backend may be given 2 optional
        arguments at construction:
            - ``fs_db_path`` (DEFAULT: "SQLiteVCDStore.db")
                - A path to the feature store database file. This may be
                  relative and is interpreted relative to the given/default
                  ``db_root`` path.
            - ``db_root`` (DEFAULT: the current working directory)

        :param _backend: The backend instance that defined actual frame store
            implementation and behavior.
        :type _backend: type
        :param args: Positional arguments to pass to the given backend.
        :param kwargs: Keyword arguments to pass to the given backend
        implementation type upon construction.

        """
        if '_backend' in kwargs:
            backend_type = kwargs.get('_backend')
            assert issubclass(backend_type, VCDStoreBackendInterface), \
                "Provided backend was not a subclass of the standard interface!"
            kwargs.pop('_backend')
            self._backend = backend_type(*args, **kwargs)
        else:
            self._backend = SQLiteVCDStoreBackend(*args, **kwargs)

    def store_feature(self, feature_store_elements, overwrite=False):
        """
        Store one or more VCDStoreElement objects.

        :param feature_store_elements: list of feature store element to store.
        :type feature_store_elements: VCDStoreElement or list of VCDStoreElement
        :param overwrite: If a feature already exists at the provides key
            location, overwrite that feature with the provided feature,
            preventing an exception from being thrown.
        :type overwrite: bool

        :raise VCDDuplicateFeatureError: If a feature already exists for the given
            video id.


        """
        self._backend.store_feature(feature_store_elements, overwrite=overwrite)

    def get_feature(self, descriptor_id, video_id, frame_num=None,
                    timestamp=None, spacial_x=None, spacial_y=None):
        """
        Query a single VCDStoreElement that exactly matches the given metadata.
        There should never exist multiple features for the same set of metadata.
        A feature may not exist for the given set of metadata, raising an
        exception.

        :param descriptor_id: The id string of the descriptor from where this
            feature vector is coming from.
        :type descriptor_id: str
        :param video_id: The ID key of the video from which this feature vector
            was constructed.
        :type video_id: int
        :param frame_num: The frame number of the feature to retrieve. This is
            optional.
        :type frame_num: int or None
        :param timestamp: The UNIX timestamp of the feature to retrieve. This is
            optional.
        :type timestamp: float or None
        :param spacial_x: The spacial X value of the feature to retrieve. This
            is optional.
        :type spacial_x: float
        :param spacial_y: The spacial Y value of the feature to retrieve. This
            is optional.
        :type spacial_y: float

        :raise VCDNoFeatureError: If no feature exists for the designated key(s).

        :return: The VCDStoreElement matching the query.
        :rtype: VCDStoreElement

        """
        return self._backend.get_feature(descriptor_id=descriptor_id,
                                         video_id=video_id,
                                         frame_num=frame_num,
                                         timestamp=timestamp,
                                         spacial_x=spacial_x,
                                         spacial_y=spacial_y)

    def get_features_by(self, descriptor_id=None, video_id=None, frame_num=None,
                        timestamp=None, spacial_x=None, spacial_y=None):
        """
        Retrieve a tuple of VCDStoreElement objects that all, or in part, match
        the specified criteria. I.e. providing no criteria (no parameters) will
        fetch all features (not particularly recommended), or just providing a
        descriptor id, a video id and a spacial x value will fetch all features
        produced by the given descriptor on the given video with the given
        spacial x value.

        :param descriptor_id: The descriptor ID that produced the feature.
        :type descriptor_id: str
        :param video_id: The video ID that the feature was computed on.
        :type video_id: int
        :param frame_num: The frame number that the feature was computed on.
        :type frame_num: int
        :param timestamp: The timestamp of the image data the feature
            represents.
        :type timestamp: float
        :param spacial_x: The spacial X value the feature describes.
        :type spacial_x: float
        :param spacial_y: The spacial Y value the feature describes.
        :type spacial_y: float

        :return: A tuple of matching VCDStoreElement objects that match the
            query. This may be an empty tuple if there were no matching results.
        :rtype: tuple of VCDStoreElement

        """
        return self._backend.get_features_by(descriptor_id, video_id, frame_num,
                                             timestamp, spacial_x, spacial_y)
