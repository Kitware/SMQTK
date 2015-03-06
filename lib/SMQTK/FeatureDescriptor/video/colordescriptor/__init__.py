"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import os
import os.path as osp

from SMQTK.FeatureDescriptor import FeatureDescriptor


class ColorDescriptor_CSIFT (FeatureDescriptor):
    """
    CSIFT colordescriptor feature descriptor
    """

    def compute_feature(self, data):
        """
        Compute CSIFT colordescriptor feature given a VideoFile instance.

        :param data: Video file wrapper
        :type data: SMQTK.utils.VideoFile.VideoFile

        :return: Video feature vector
        :rtype: numpy.ndarray

        """
        self.log.info("Processing video: %s", data.filepath)

        # Creating subdirectory to put video-specific work files in
        video_work_dir = osp.join(self.work_directory,
                                  *data.split_md5sum(8))
        if not osp.isdir(video_work_dir):
            os.makedirs(video_work_dir)

        # Pre-define key working file paths
        work_combined_features = osp.join(video_work_dir,
                                          "csift.combined.txt")

        ###
        # Create combined feature file from per-frame features
        # Only extracting every 2 seconds to get sparse representation
        #
        frame_list = data.frame_map(0, 2)
