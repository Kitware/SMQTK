"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging


class VCDStoreBackendInterface(object):
    """
    Interface class for creating a backbone for use in a VCDStore object.

    This class must override the following methods. More information about each
    method can be found in the doc string in the interface method definitions.
    No super call should be made, as these interface-level methods will raise a
    NotImplementedError.

        store_feature(...)
        get_feature(...)
        get_features_by(...)

    All methods in this interface are set to raise a NotImplementedError, so
    don't super call them (else you will get that error).

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        """
        Interface class for creating a backbone for use in a VCDStore object.
        This object will raise NotImplementedError as this class should never be
        instantiated directly.

        Initializes a logging object, '_log', extending the namespace of the
        module with the name of the implementation class's name.

        @raise NotImplementedError: If the FeatureStoreBackendInterface is
        instantiated directly.

        """
        self._log = logging.getLogger('.'.join([__name__,
                                                self.__class__.__name__]))

    @abc.abstractmethod
    def store_feature(self, feature_elements, overwrite=False):
        """
        Store one or more VCDStoreElement objects.

        :param feature_elements: list of feature store element to store.
        :type feature_elements: VCDStoreElement or list of VCDStoreElement
        :param overwrite: If a feature already exists at the provides key
            location, overwrite that feature with the provided feature,
            preventing an exception from being thrown.
        :type overwrite: bool

        :raise VCDDuplicateFeatureError: If a feature already exists for the given
            video id.

        """
        return

    @abc.abstractmethod
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
        return NotImplemented

    @abc.abstractmethod
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
        return ()
