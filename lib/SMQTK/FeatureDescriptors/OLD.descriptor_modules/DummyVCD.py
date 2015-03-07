"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy as np

from ...VCDStore.VCDStoreElement import VCDStoreElement
from .. import VCDWorkerInterface


class DummyVCD (VCDWorkerInterface):
    """
    Dummy VCD worker
    """

    DESCRIPTOR_ID = "Dummy"

    def __init__(self, config, working_dir, image_root=None):
        super(DummyVCD, self).__init__(config, working_dir, image_root)

    def process_video(self, video_file):
        did = self.DESCRIPTOR_ID
        _, vid = self.get_video_prefix(video_file)
        feat = np.array((1, 2, 3, 4, 5))

        return VCDStoreElement(did, int(vid), feat)

WORKER_CLASS = DummyVCD
