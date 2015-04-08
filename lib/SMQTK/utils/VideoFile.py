"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import imageio
import numpy
import os
import os.path as osp
import re
import shutil
import StringIO
import subprocess
import time

from SMQTK.utils import DataFile, safe_create_dir


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
        self.__preview_cache = None

    def get_preview_image(self, save_dir=None, regenerate=False):
        """
        Generate and return a preview GIF animation for this video file. File
        saved is named according to the format: %s.preview.gif, where the '%s'
        is the MD5 hex sum of the video that the preview is of.

        If a preview has already been generated somewhere and still exists on
        the file system, we simply return the cached path to that file, if
        regenerate is False.

        :param save_dir: Optional directory to save generated GIF image file to.
            By default we save it in this video file's working directory.
        :type save_dir: str

        :param regenerate: Force regeneration of the preview GIF image file.
            This also rewrites the cached location so the regenerated file is
            new returned.
        :type regenerate: bool

        :return: The path to a preview image for this data file.
        :rtype: str

        """
        if (self.__preview_cache is None
                or not osp.isfile(self.__preview_cache)
                or regenerate):
            self.log.debug("[%s] Populating preview GIF cache", self)
            fname = "%s.preview.gif" % self.md5sum
            if save_dir:
                safe_create_dir(save_dir)
                target_fp = osp.join(save_dir, fname)
            else:
                target_fp = osp.join(self.work_directory, fname)
            self.log.debug("[%s] Preview image file: %s", self, target_fp)
            # if the file already exists, we don't need to generate it again
            if not osp.isfile(target_fp):
                self.log.debug("[%s] Extracting frames for GIF", self)
                md = self.metadata()
                offset = md.duration * 0.2
                interval = 0.5  # ~2fps gif
                max_duration = min(10.0, md.duration * 0.6)
                fm = self.frame_map(offset, interval, max_duration)
                self.log.debug("[%s] GIF file doesn't exist, generating", self)
                img_arrays = []
                for frm_num in sorted(fm.keys()):
                    img_arrays.append(imageio.imread(fm[frm_num]))
                imageio.mimwrite(target_fp, img_arrays, duration=interval)
                self.log.debug("[%s] Finished generating GIF", self)
            else:
                self.log.debug("[%s] Already exists", self)
            self.__preview_cache = target_fp

        return self.__preview_cache

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
            PROC_FFPROBE = "ffprobe"
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
        :type second_interval: float

        :param max_duration: Maximum number of seconds worth of extracted frames
        :type max_duration: float

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
            self.log.debug("Only extracting specified frames: %s", frames)
            extract_indices.update(frames)
        else:
            self.log.debug("Determining frames needed for specification: "
                           "offset: %f, interval: %f, max_duration: %f",
                           second_offset, second_interval, max_duration)
            extract_indices.update(
                self._get_frames_for_interval(num_frames, video_md.fps,
                                              second_offset, second_interval,
                                              max_duration)
            )

        if not extract_indices:
            return {}

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
                                 max_duration=0.0):
        """
        Return a tuple of frame numbers from the given number of frames
        specification, taking into account the give time offset, time interval
        and maximum duration. All time parameters should be in seconds. Indices
        returned are 0-based (i.e. first frame is 0, not 1).

        We are making a sensible assumption that we are not dealing with frame
        speeds of over 1000Hz and rounding frame frame times to the neared
        thousandth of a second to mitigate floating point error.

        :param num_frames: Number of frame numbers to consider. This should be,
            at most, the total number of frames in the video.
        :param fps: The FPS rating of the video.
        :param offset: The number of seconds into the video to start gathering
            frames.
        :param interval: The minimum second interval between frames taken.
        :param max_duration: Maximum second duration of collected frames

        :rtype: list of int

        """
        # Interpolating based i interval
        # self.log.debug("Starting frame gathering...")
        # self.log.debug("-- num_frames: %s", num_frames)
        # self.log.debug("-- fps: %s", fps)
        # self.log.debug("-- offset: %s", offset)
        # self.log.debug("-- interval: %s", interval)
        # self.log.debug("-- max duration: %s", max_duration)
        fps = float(fps)
        first_frame = offset * fps
        self.log.debug("First Frame: %f", first_frame)
        if max_duration:
            cutoff_frame = min(num_frames, (max_duration + offset) * fps)
        else:
            cutoff_frame = float(num_frames)
        self.log.debug("Cutoff frame: %f", cutoff_frame)
        if interval:
            incr = interval * fps
        else:
            incr = 1
        self.log.debug("Frame increment: %f", incr)

        # Interpolate
        frms = [first_frame]
        next_frm = first_frame + incr
        while next_frm < cutoff_frame:
            self.log.debug("-- adding frame: %f", next_frm)
            frms.append(next_frm)
            next_frm += incr
        # in-place cast all elements to int
        self.log.debug("int casting frames...")
        for i in xrange(len(frms)):
            frms[i] = int(frms[i])
        return frms

    def _extract_frames(self, frame_list, output_ext):
        """
        Extract specific frames from our configured video file.

        This function is called within a locked section of code (filesystem
        based).

        :param frame_list: List of frame numbers to extract. Must be sorted.
        :type frame_list: list of int

        :param output_ext: Output file extension
        :type output_ext: str

        """
        # TODO: In the future, this should be abstract and left to a subclass to
        #       implement.

        # Setup temp extraction directory
        tmp_extraction_dir = osp.join(self.work_directory, ".TMP")
        if osp.isdir(tmp_extraction_dir):
            self.log.debug("Existing temp director found, removing and "
                           "starting over")
            shutil.rmtree(tmp_extraction_dir, ignore_errors=True)
        os.makedirs(tmp_extraction_dir)

        md = self.metadata()
        frame_times = numpy.array(frame_list) / md.fps
        sPIPE = subprocess.PIPE
        # TODO: thread/multiprocess this.
        for f, t in zip(frame_list, frame_times):
            cmd = ['ffmpeg', '-accurate_seek', '-ss', str(t),
                   '-i', self.filepath,
                   '-frames:v', '1',
                   osp.join(tmp_extraction_dir, "%08d.%s" % (f, output_ext))]
            self.log.debug("Frame extraction command: %s", cmd)

            p = subprocess.Popen(cmd, stdout=sPIPE, stderr=sPIPE)
            _, _ = p.communicate()
            if p.returncode != 0:
                raise RuntimeError("FFmpeg failed to extract frame at time %f! "
                                   "(return code: %d)"
                                   % (t, p.returncode))

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
