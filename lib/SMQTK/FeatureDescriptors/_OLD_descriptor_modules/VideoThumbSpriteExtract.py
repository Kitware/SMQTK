"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from .. import VCDWorkerInterface


class VideoThumbSpriteExtract (VCDWorkerInterface):
    """
    Dummy VCD worker which extracts the frames and uploads them to the server
    """

    DESCRIPTOR_ID = "VideoThumbSpriteExtract"

    def __init__(self, config, working_dir, image_root):
        super(VideoThumbSpriteExtract, self).__init__(config, working_dir,
                                                      image_root)

    def process_video(self, video_file):
        try:
            self.mp4_extract_video_frames(video_file)
        except:
            # TODO: This should become a more specific except clause
            #       when/if used in production
            self._log.info("Frames extraction job failed")

    @classmethod
    def generate_config(cls, config=None):
        config = super(VideoThumbSpriteExtract, cls).generate_config(config)
        return config

WORKER_CLASS = VideoThumbSpriteExtract
