"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os.path as osp
from SMQTK.utils import DataIngest, VideoFile, safe_create_dir


class VideoIngest (DataIngest):
    """
    Ingest of VideoFile type elements.
    """

    def DATA_FILE_TYPE(self, filepath, uid=None, md5_shortcut=None):
        """
        VideoIngest data file factory method.

        :param filepath: Path to the data file
        :type filepath: str

        :param uid: Optional UID of the item
        :type uid: int

        :param md5_shortcut: MD5 hexdigest string, if its already known, to use
            instead of needing to explicitly compute it. This saves processing
            time.
        :type md5_shortcut: str

        :return: New VideoFile instance
        :rtype: VideoFile

        """
        # Figure out the work directory for this video file in the ingest's work
        # space
        d = super(VideoIngest, self).DATA_FILE_TYPE(filepath, uid=uid,
                                                    md5_shortcut=md5_shortcut)
        md5_split = d.split_md5sum(8)
        v_work_dir = osp.join(self.work_directory, *md5_split)
        vf = VideoFile(filepath, v_work_dir, uid)
        # Punch in MD5 sum that we already spent the time computing and don't
        # want to waste
        vf._md5_cache = ''.join(md5_split)
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
        vf_preview_dir = osp.join(self.previews_directory,
                                  *data.split_md5sum(8)[:-1])
        if not osp.isfile(data.get_preview_image(vf_preview_dir)):
            raise RuntimeError("Failed to generate GIF preview for video file "
                               "%s" % data)
