"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy as np


class VCDStoreElement(object):
    """
    Encapsulation of a descriptor's generated feature vector and its associated
    metadata. All numerical values must be non-negative
    """

    def __init__(self, descriptor_id, video_id, feat_vec,
                 frame_num=None, timestamp=None, spacial_x=None,
                 spacial_y=None):
        """
        Construct a VCDStoreElement

        :param descriptor_id: Descriptor that this feature was generated from
        :type descriptor_id: str
        :param video_id: ID of the video that this feature was generated from
        :type video_id: int
        :param feat_vec: The numpy.ndarray array feature vector.
        :type feat_vec: numpy.ndarray
        :param frame_num: Frame number this feature was generated from.
        :type frame_num: int or None
        :param timestamp: UNIX timestamp related to this feature's generation
        :type timestamp: float or None
        :param spacial_x: Spacial X value this feature was generated from.
        :type spacial_x: float or None
        :param spacial_y: Special Y value this feature was generated from.
        :type spacial_y: float or None

        """
        self.descriptor_id = str(descriptor_id)
        self.video_id = int(video_id)
        assert isinstance(feat_vec, np.ndarray) and len(feat_vec.shape) == 1, \
            "Not given a numpy.ndarray or it was not flat. Flatness is " \
            "important because we cannot reconstruct the structure from a " \
            "buffer into anything more than that."
        self.feat_vec = feat_vec
        self.frame_num = int(frame_num) \
            if frame_num is not None and int(frame_num) >= 0 else None
        self.timestamp = float(timestamp) \
            if timestamp is not None and float(timestamp) >= 0 else None
        self.spacial_x = float(spacial_x) \
            if spacial_x is not None and float(spacial_x) >= 0 else None
        self.spacial_y = float(spacial_y) \
            if spacial_y is not None and float(spacial_y) >= 0 else None

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return "VCDStoreElement%s" % self.__repr__()

    def __eq__(self, other):
        if (isinstance(other, VCDStoreElement)
                and self.descriptor_id == other.descriptor_id
                and self.video_id == other.video_id
                and all(self.feat_vec == other.feat_vec)
                and self.frame_num == other.frame_num
                and self.timestamp == other.timestamp
                and self.spacial_x == other.spacial_x
                and self.spacial_y == other.spacial_y):
            return True
        return False

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((
            self.descriptor_id, self.video_id, self.feat_vec,
            self.frame_num, self.timestamp,
            self.spacial_x, self.spacial_y
        ))
