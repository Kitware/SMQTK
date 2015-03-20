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

    def _register_data_item(self, data):
        """ Internal add-data-to-maps function
        :param data: DataFile instance to add.
        :type data: VideoFile
        """
        super(VideoIngest, self)._register_data_item(data)

        # Generate preview GIF if needed by calling preview getter the
        # data-based location
        self.log.debug("Generating preview GIF for video file...")
        vf_preview_dir = osp.join(self.previews_directory,
                                  *data.split_md5sum(8)[:-1])
        if not osp.isfile(data.get_preview_image(vf_preview_dir)):
            raise RuntimeError("Failed to generate GIF preview for video file "
                               "%s" % data)
