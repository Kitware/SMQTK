"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from SMQTK.utils import DataIngest, VideoFile


class VideoIngest (DataIngest):
    """
    Ingest of video files
    """

    def DATA_FILE_TYPE(self, filepath):
        return VideoFile(filepath, self._base_work_dir)

    def __init__(self, name, base_data_dir, base_work_dir, starting_index=0):
        self._base_work_dir = base_work_dir
        super(VideoIngest, self).__init__(name, base_data_dir, base_work_dir,
                                          starting_index)
