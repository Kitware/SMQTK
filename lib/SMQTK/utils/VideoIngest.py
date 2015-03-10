"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os
from SMQTK.utils import DataFile, DataIngest, VideoFile


class VideoIngest (DataIngest):
    """
    Ingest of VideoFile type elements.
    """

    def DATA_FILE_TYPE(self, filepath, uid=None):
        # Figure out the work directory for this video file in the ingest's work
        # space
        md5_split = DataFile(filepath).split_md5sum(8)
        md5 = ''.join(md5_split)
        v_work_dir = os.path.join(self.work_directory, *md5_split)
        vf = VideoFile(filepath, v_work_dir, uid)
        # Punch in MD5 sum that we already spent the time computing and don't
        # want to waste
        vf._DataFile__md5_cache = md5
        return vf
