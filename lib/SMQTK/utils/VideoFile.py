"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import os
import os.path as osp
import re
import shutil
import subprocess
import time

from SMQTK.utils import DataFile


class VideoMetadata (object):
    """
    Container for simple video file metadata values
    """

    def __init__(self):
        #: :type: None or int
        self.width = None
        #: :type: None or int
        self.height = None
        #: :type: None or float
        self.fps = None
        #: :type: None or float
        self.duration = None


class VideoFile (DataFile):
    """
    Class representing an video file with methods for getting the video as well
    as extracting and retrieving video frames.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath, work_directory, uid=None):
        super(VideoFile, self).__init__(filepath, uid)
        self._work_dir = work_directory

        # Cache variables
        self.__metadata_cache = None

    def get_preview_image(self):
        """
        :return: The path to a preview image for this data file.
        :rtype: str
        """
        # TODO: Generate a GIF sequence within some simple interval
        # For now, just returning a frame 20% into the video
        md = self.metadata()
        perc20 = int(md.duration * md.fps * 0.2)
        self.log.debug("Extracting preview frame %d", perc20)
        return self.frame_map(frames=[perc20])[perc20]

    @property
    def work_directory(self):
        """
        Work directory for this video. Generally, this is the directory where
        extracted frames will be located.

        :return: A path
        :rtype: str

        """
        if not os.path.isdir(self._work_dir):
            os.makedirs(self._work_dir)
        return self._work_dir

    def metadata(self):
        """
        :return: the simple metadata for this video
        :rtype: VideoMetadata
        """
        # TODO: In the future, this should be abstract and left to a subclass to
        #       implement.

        if self.__metadata_cache is None:
            PROC_FFPROBE = "smqtk_ffprobe"
            re_float_match = "[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?"

            self.log.debug("Using ffprobe: %s", PROC_FFPROBE)
            cmd = [PROC_FFPROBE, '-i', self.filepath]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            out, err = p.communicate()
            # ffprobe puts output to err stream
            if p.returncode:  # non-zero
                raise RuntimeError("Failed to probe video file. Error:\n%s"
                                   % err)

            # WxH
            m = re.search("Stream.*Video.* (\d+)x(\d+)", err)
            if m:
                width = int(m.group(1))
                height = int(m.group(2))
            else:
                raise RuntimeError("Couldn't find width/height specification "
                                   "for video file '%s'" % self.filepath)

            # FPS
            m = re.search("Stream.*Video.* (%s) fps" % re_float_match, err)
            if m:
                fps = float(m.group(1))
            else:
                # falling back on tbr measurement
                self.log.debug("Couldn't find fps measurement, looking for TBR")
                m = re.search("Stream.*Video.* (%s) tbr" % re_float_match, err)
                if m:
                    fps = float(m.group(1))
                else:
                    raise RuntimeError("Couldn't find tbr specification for "
                                       "video file '%s'" % self.filepath)

            # Duration
            m = re.search("Duration: (\d+):(\d+):(%s)" % re_float_match, err)
            if m:
                duration = (
                    (60 * 60 * int(m.group(1)))     # hours
                    + (60 * int(m.group(2)))        # minutes
                    + float(m.group(3))             # seconds
                )
            else:
                raise RuntimeError("Couldn't find duration specification for "
                                   "video file '%s'" % self.filepath)

            md = VideoMetadata()
            md.width = width
            md.height = height
            md.fps = fps
            md.duration = duration
            self.__metadata_cache = md

        return self.__metadata_cache

    def frame_map(self, second_offset=0, second_interval=0, max_duration=0,
                  frames=(), output_image_ext='png'):
        """
        Return a map of video frame index to image file in the given format.

        If frames request have not yet been extracted, they are done now. This
        means that this method could take a little time to complete if there
        are many frames in the video file and this is the first time this is
        being called.

        This may return an empty list if there are no frames in the video for
        the specified, or default, constraints.

        :param second_offset: Seconds into the video to start extracting
        :type second_offset: float

        :param second_interval: Number of seconds between extracted frames
        :type second_offset: float

        :param max_duration: Maximum number of seconds worth of extracted frames
        :type second_offset: float

        :param frames: Specific exact frames within the video to extract.
            Providing explicit frames causes other parameters to be ignored and
            only the frames specified here to be extracted and returned.
        :type frames: list of int

        :return: Map of frame-to-filepath for requested video frames
        :rtype: dict of (int, str)

        """
        frame_dir = self.work_directory
        video_md = self.metadata()

        # Frames to extract from video
        num_frames = int(video_md.fps * video_md.duration)
        extract_indices = set()
        if frames:
            extract_indices.update(frames)
        else:
            extract_indices.update(
                self._get_frames_for_interval(num_frames, video_md.fps,
                                              second_offset, second_interval,
                                              max_duration)
            )

        if not extract_indices:
            return []

        # frame/filename map that will be returned based on requested frames
        frame_map = dict(
            (i, osp.join(frame_dir, self._get_file_name(i, output_image_ext)))
            for i in extract_indices
        )

        ###
        # Check to see if another process is working on this video via a lock
        # file. If lock file present, wait until other process has completed
        # before determining what we need to produce, as the other process may
        # be trying to extract the same frames as us (don't want to duplicate
        # work).
        #
        # NOTE: This is prone to starvation if tons of processes are trying
        #       extract the same video frames, but this not probable due to use
        #       case.
        #
        lock_file = osp.join(frame_dir, '.lock')
        self.log.debug("Acquiring file lock...")
        while not self._exclusive_touch(lock_file):
            time.sleep(0.01)
        self.log.debug("--> Lock acquired!")

        try:
            ###
            # Determine frames to extract from existing files (if any)
            #
            frames_to_process = []
            for i, img_file in sorted(frame_map.items()):
                # If file doesn't exist, add it to the frame-to-file map,
                # indicating that it needs processing.
                if not osp.isfile(img_file):
                    self.log.debug('frame %d needs processing', i)
                    frames_to_process.append(i)

            ###
            # Extract needed frames via hook function that provides
            # implementation.
            #
            if frames_to_process:
                self._extract_frames(frames_to_process, output_image_ext)

            return frame_map
        finally:
            os.remove(lock_file)

    @staticmethod
    def _get_file_name(frame, ext):
        """ Get a standard filename for a frame number
        :param frame: frame number
        :type frame: int
        :return: file name
        :rtype:str
        """
        return "%08d.%s" % (frame, ext)

    @staticmethod
    def _exclusive_touch(file_path):
        """
        Attempt to touch a file. If that file already exists, we return False.
        If the file was touched and created, we return True. Other OSErrors
        thrown beside the expected "file already exists" error are passed
        upwards.

        :param file_path: Path to the file to touch.
        :type file_path: str

        :return: True if we touched/created the file, false if we couldn't
        :rtype: bool

        """
        try:
            fd = os.open(file_path, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            return True
        except OSError, ex:
            if ex.errno == 17:
                return False
            else:
                raise

    def _get_frames_for_interval(self, num_frames, fps, offset, interval,
                                 max_duration=0):
        """
        Return a tuple of frame numbers from the given number of frames
        specification, taking into account the give time offset, time interval
        and maximum duration. All time parameters should be in seconds. Indices
        returned are 0-based (i.e. first frame is 0, not 1).

        We are making a sensible assumption that we are not dealing with frame
        speeds of over 1000Hz and rounding frame frame times to the neared
        thousandth of a second to mitigate floating point error.

        :rtype: list of int

        """
        frames_taken = []

        # run through frames, counting the seconds per frame. when time interval
        # elapses, take the last frame encountered.
        cur_time = 0.0
        frame_time_interval = 1.0 / fps
        next_threshold = None

        # self.log.debug("Starting frame gathering...")
        # self.log.debug("-- num_frames: %s", num_frames)
        # self.log.debug("-- fps: %s", fps)
        # self.log.debug("-- offset: %s", offset)
        # self.log.debug("-- interval: %s", interval)
        # self.log.debug("-- max duration: %s", max_duration)
        for frame in xrange(num_frames):
            # self.log.debug("Frame >> %d", frame)
            # self.log.debug("... Cur frame time: %.16f -> %.3f",
            #                cur_time, round(cur_time, 3))

            # If we has surpassed our given maximum duration (taking the offset
            # into account), kick out
            if max_duration and (cur_time - offset) >= max_duration:
                break

            if round(cur_time, 3) >= round(offset, 3):
                if frames_taken:
                    if round(cur_time, 3) >= round(next_threshold, 3):
                        # self.log.debug("... T exceeded, gathering frame")
                        # take frame, set next threshold
                        frames_taken.append(frame)
                        next_threshold += interval
                else:
                    # self.log.debug("... First frame")
                    # first valid frame seen, this is our starting frame
                    frames_taken.append(frame)
                    next_threshold = cur_time + interval

                # self.log.debug("... Next T: %.16f -> %.3f",
                #                next_threshold, round(next_threshold, 3))

            cur_time += frame_time_interval

        return frames_taken

    def _extract_frames(self, frame_list, output_ext):
        """
        Extract specific frames from our configured video file.

        This function is called with a locked section of code.

        :param frame_list: List of frame numbers to extract. Must be sorted.
        :type frame_list: list of int

        :param output_ext: Output file extension
        :type output_ext: str

        """
        # TODO: In the future, this should be abstract and left to a subclass to
        #       implement.

        PROC_FRAME_EXTRACTOR = "smqtk_frame_extractor"

        # Setup temp extraction directory
        tmp_extraction_dir = osp.join(self.work_directory, ".TMP")
        if osp.isdir(tmp_extraction_dir):
            self.log.debug("Existing temp director found, removing and "
                           "starting over")
            shutil.rmtree(tmp_extraction_dir, ignore_errors=True)
        os.makedirs(tmp_extraction_dir)

        str_frame_selection = ','.join(map(str, frame_list))

        cmd = [PROC_FRAME_EXTRACTOR,
               '-i', self.filepath,
               '-f', str_frame_selection,
               '-o', osp.join(tmp_extraction_dir, "%%08d.%s" % output_ext)]
        self.log.debug("Extractor command: %s", cmd)

        ret_code = subprocess.call(cmd)
        if ret_code != 0:
            raise RuntimeError("Frame extractor utility failed! (return code: "
                               "%d)" % ret_code)
        else:
            generated_files = [osp.join(tmp_extraction_dir, f)
                               for f in sorted(os.listdir(tmp_extraction_dir))]

            # If there is a mismatch in number of frames requested and number of
            # frames produced, something went wrong
            if len(generated_files) != len(frame_list):
                raise RuntimeError("Failed to extract all frames requested!")

            # Move generated files out of temp directory
            for fn, gf in zip(frame_list, generated_files):
                os.rename(gf, osp.join(self.work_directory,
                                       self._get_file_name(fn, output_ext)))
            os.removedirs(tmp_extraction_dir)
            self.log.debug("Frame extraction complete")
