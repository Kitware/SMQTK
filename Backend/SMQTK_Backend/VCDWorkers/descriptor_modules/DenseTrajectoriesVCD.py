"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy as np
import os.path as osp
import subprocess
import time

from .. import VCDWorkerInterface
from ...VCDStore import VCDStore
from ...VCDStore.VCDStoreElement import VCDStoreElement
from ...VCDStore.errors import VCDNoFeatureError


# TODO: Assign this to the class below when it has been updated.
WORKER_CLASS = None


# TODO: This class needs updating to work with current code base
class DenseTrajectories (VCDWorkerInterface):
    """ Execution Wrapper for the dense trajectories descriptor

    (http://lear.inrialpes.fr/people/wang/dense_trajectories)

    """

    DESCRIPTOR_ID = "dense_trajectories"
    DT_EXE_PATH = 'DenseTrack'

    @classmethod
    def generate_config(cls, config=None):
        """
        Generate, define and return a configuration object for this descriptor
        worker.

        :param config: An optionally existing configuration object to update.
        :type config: None or SafeConfigCommentParser
        :return: Updated configuration object with this descriptor's specific
            configuration parameters.
        :rtype: SafeConfigCommentParser

        """
        config = super(DenseTrajectories, cls).generate_config(config)

        sect = cls.DESCRIPTOR_ID
        if not config.has_section(sect):
            config.add_section(sect)

        config.set(sect, "dense_track_exe", '',
                   "Path to the DenseTrack executable to use.")

        return config

    def __init__(self, video_file_path):
        """ Initialize dense trajectories worker

        Requires the path to the executable to run.

        """
        super(DenseTrajectories, self).__init__(video_file_path)

        self._exe_path = osp.expanduser(self.DT_EXE_PATH)
        self._dt_intermdiate_store = \
            VCDStore(fs_db_path='dt_intermediate.db')

    def process_video(self, video_file):
        """ Run the dense trajectories feature detector on the given video

        This descriptor produces per-frame spacial feature data. Data will be
        stored for each frame (Nx433 shape matrix of all trajectory features/
        tracks that ended on that frame), as well as by spacial coordinates.

        Spacial coordinates are determined be normalizing the trajectory's
        x and y means by the video's width and height respectively, bringing
        the spacial location in between [0,1].

        @raise RuntimeError: Failed to successfully run executable (non-zero
        return code observed).

        """
        pfx, video_id = self.get_video_prefix(video_file)

        # TODO: This method needs updating from a previous version.
        #       Mainly, remove feature store usage from here, returning the
        #       store elements only.

        # Try to get some features from the VCDStore from this video. If
        # there is a video-level entry already present, don't process anything.
        try:
            self._log.info("[%i] Checking for complete feature for video",
                           video_id)
            self._feature_store.get_feature(self.DESCRIPTOR_ID,
                                            video_id)
            # A returned feature means that we've already processed this video.
            self._log.info("[%i] Feature found. Skipping video.", video_id)
            return
        except VCDNoFeatureError:
            self._log.info('[%i] No video-level features stored. Continuing '
                           'with processing.', video_id)

        # Run executable, pushing feature output to the temp file.
        args = [self._exe_path, video_file]
        self._log.debug("[%i] running: %s", video_id, args)
        self._log.info("[%i] starting Dense Trajectories executable: %s",
                       video_id, self._exe_path)
        exe_start = time.time()
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        txt_out = p.communicate()[0]
        rc = p.returncode
        if rc != 0:
            self._log.error("[%i] Call to executable resulted in non-0 return "
                            "code", video_id)
            raise RuntimeError('Call to executable resulted in non-0 return '
                               'code')
        self._log.info("[%i] DenseTrack execution time: %f",
                       video_id, time.time() - exe_start)

        # take text output and create numpy matrix. Text is read in as a
        # 1-dimensional array. Reshaping to known shape: [N x 433]
        # -> specifically 433 because of the default values DenseTrack uses
        self._log.debug('[%i] loading produced features as numpy matrix',
                        video_id)
        load_start = time.time()
        a = np.fromstring(txt_out, dtype=float, sep='\t')
        try:
            features = np.reshape(a, (a.size / 433, 433))
        except ValueError:
            self._log.error("[%i] No features calculated for video %s",
                            video_id, self._video_file_path)
            return
        self._log.info("[%i] text to numpy load time: %f",
                       video_id, time.time() - load_start)

        # figure out video information
        v_metadata = self._introspect_video()
        v_width = float(v_metadata['width'])
        v_height = float(v_metadata['height'])

        construct_start = time.time()
        feature_store_elements = list()
        for row in features:
            frame_num = int(row[0])
            mean_x = row[1]
            mean_y = row[2]
            spacial_x = mean_x / v_width
            spacial_y = mean_y / v_height
            fs_element = VCDStoreElement(descriptor_id=self.DESCRIPTOR_ID,
                                         video_id=video_id,
                                         feat_vec=row,
                                         frame_num=frame_num,
                                         spacial_x=spacial_x,
                                         spacial_y=spacial_y)
            feature_store_elements.append(fs_element)
            # add things to the collection lists for this per-spacial entry
            self._log.debug('collecting %s', (frame_num, spacial_x, spacial_y))
        self._log.info("[%i] VCDStoreElement construction time: %f",
                       video_id, time.time() - construct_start)

        # perform batch store
        self._log.info("[%i] storing intermediate features (size: %i)",
                       video_id, features.shape[0])
        store_start = time.time()
        self._dt_intermdiate_store.store_feature(feature_store_elements)
        self._log.info("[%i] intermediate storage time: %f",
                       video_id, time.time() - store_start)
