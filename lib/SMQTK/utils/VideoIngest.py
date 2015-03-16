"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os.path as osp
from SMQTK.utils import DataFile, DataIngest, VideoFile, safe_create_dir


class VideoIngest (DataIngest):
    """
    Ingest of VideoFile type elements.
    """

    def DATA_FILE_TYPE(self, filepath, uid=None):
        # Figure out the work directory for this video file in the ingest's work
        # space
        md5_split = DataFile(filepath).split_md5sum(8)
        md5 = ''.join(md5_split)
        v_work_dir = osp.join(self.work_directory, *md5_split)
        vf = VideoFile(filepath, v_work_dir, uid)
        # Punch in MD5 sum that we already spent the time computing and don't
        # want to waste
        vf._DataFile__md5_cache = md5
        return vf

    @property
    def previews_directory(self):
        """
        :return: Sub-directory under data directory that contains preview GIF
            files for each ingested video file.
        :rtype: str
        """
        d = osp.join(self.data_directory, "previews")
        if not osp.isdir(d):
            safe_create_dir(d)
        return d

    def add_data_file(self, origin_filepath):
        """
        Add the given data file to this ingest

        The original file is copied and further maintenance of the original
        file is left to the user.

        If the given file exists in the ingest already, we do not add a second
        copy, instead returning the DataFile instance of the existing. Check max
        UID before and after this call to check for new ingest file or not.

        As this is a video ingest, we also compute the preview image GIF file
        upon ingest.

        :param origin_filepath: Path to a file that should be added to this
            ingest.
        :type origin_filepath: str

        :return: The DataFile instance that was just ingested
        :rtype: VideoFile

        """
        vf = super(VideoIngest, self).add_data_file(origin_filepath)

        # Generate preview by calling preview getter with regenerate flag,
        # forcing generation to the given cache location in the data directory.
        vf_preview_dir = osp.join(self.previews_directory,
                                  *vf.split_md5sum(8)[:-1])
