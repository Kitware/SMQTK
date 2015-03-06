"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import os
import os.path as osp
import re
import shutil
import subprocess
import time

from SMQTK_Backend.utils import SafeConfigCommentParser


# TODO: Make this descend from ControllerProcess to add in asynchronous
#       processing capability outside of the MPI framework.
class VCDWorkerInterface (object):
    """ Abstract base class defining API and support methods for a descriptor
    worker implementation.

    Descriptor workers should take in a video file (.mp4 or the like) and
    produce feature data, storing VCDStoreElement structures.

    """
    __metaclass__ = abc.ABCMeta

    EXTRACTOR_CONFIG_SECT = 'frame_extractor'
    PROBE_CONFIG_SECT = 'video_probe'

    DESCRIPTOR_ID = None

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
        if config is None:
            config = SafeConfigCommentParser()

        # If the config sections for the utility methods are not present, add
        # them and their options.
        if not config.has_section(cls.EXTRACTOR_CONFIG_SECT):
            sect = cls.EXTRACTOR_CONFIG_SECT
            config.add_section(sect,
                               "Options controlling how frames are extracted "
                               "from videos when requested.\n"
                               "\n"
                               "The following are standard frame extraction "
                               "configurations for configurable descriptor "
                               "modules (in offset,interval,max_duration "
                               "value order):\n"
                               "    - colordescriptor -> (0, 2, 10)\n"
                               "    - subattributes   -> (3, 4, 0)\n")
            config.set(sect, 'frame_extractor_exe', 'frame_extractor',
                       "The frame_extractor executable to use")
            config.set(sect, 'start_offset_seconds', '0.0',
                       "The number of seconds after the start of the video to "
                       "start extracting frames. This may be 0 or not provided "
                       "to indicate no offset.")
            config.set(sect, 'second_interval', '2.0',
                       "Extract frames, after the provided offset, every N "
                       "seconds, where N is the value provided here. This may "
                       "be floating point. This may be 0 or not provided to "
                       "extract all frames after the given offset.")
            config.set(sect, 'max_duration_minutes', '10.0',
                       "Maximum time which we are not to extract frames past. "
                       "This should be provided in decimal minutes. This may "
                       "be not provided (no value given), or set to 0, to "
                       "indicate no maximum duration limit. ")

        if not config.has_section(cls.PROBE_CONFIG_SECT):
            sect = cls.PROBE_CONFIG_SECT
            config.add_section(sect)
            config.set(sect, 'ffprobe_exe', 'ffprobe',
                       "The ffmpeg ffprobe executable to use when probing for "
                       "video metadata.")
            config.set(sect, 'use_TBR_fallback', 'false',
                       "Use or don't use the TBR value when the FPS value is "
                       "not present when probing video metadata.")

        return config

    # TODO: Need to add a name here for unique identification.
    #       - Required for ControllerProcess constructor
    def __init__(self, config, working_dir, image_root=None):
        """
        Basic initialization, setting the working directory as well as an
        optional specification of a working image extraction output directory.
        If no image extraction directory is specified, we will define a
        directory underneath the given work directory.

        NOTE:
        Keep in mind that the work directory given is a generic working
        directory given to all descriptors. It is recommended to define a
        sub-working directory for a specific descriptor underneath this given
        directory.

        :param config: Configuration object for the descriptor worker.
        :type config: SafeConfigCommentParser
        :param working_dir: The directory where work will be stored.
        :type working_dir: str
        :param image_root: Working image output and storage root directory. If
            None is provided, a path will be determined automatically within the
            given working directory.
        :type image_root: str

        """
        if self.DESCRIPTOR_ID is None:
            raise ValueError("Sub-class did not override DESCRIPTOR_ID value "
                             "to a unique string identifier!")

        self._log = logging.getLogger('.'.join([__name__,
                                                self.__class__.__name__]))
        self.config = config
        self.working_dir = osp.abspath(working_dir)
        self.image_root_dir = image_root
        if not self.image_root_dir:
            self.image_root_dir = osp.join(self.working_dir, "extracted_frames")

    @abc.abstractmethod
    def process_video(self, video_file):
        """
        Given a video file, produce video level features, returning one or mode
        VCDStoreElement objects containing those features. If no features were
        generated, and error message should be generated, returning None.

        :param video_file: The video file to generated features for.
        :type video_file: str
        :return: Iterable of VCDStoreElement objects for this descriptor for
            features produced on the given video, or None if no features were
            generated.
        :rtype: tuple of SMQTK_Backend.VCDStore.VCDStoreElement or None

        """
        return

    # TODO: Override _run method to enable interaction with work queue for
    #       videos to process.
    #       - Return upon terminal signal.
    #       - See ECDWorkerBaseProcess for inspiration.

    @staticmethod
    def get_video_prefix(video_file):
        """
        Return the prefix and key tags for the specified video file name. The
        term "prefix" is a little misleading as the "prefix" is actually the
        last 3 digits of the numerical ID.

        Example: .../HVC998347.mp4 -> [ '347', '998347' ]

        A full or relative file path may be specified safely.

        :raises AttributeError: If file name of given file did not match the HVC
            standard.

        :param video_file: A video file path
        :type video_file: str
        :return: A tuple of the "prefix" and the full numerical ID in string
            forms. These are left in strings to maintain IDs that contain
            multiple leading 0's.
        :rtype: (str, str)

        """
        base = osp.basename(video_file)
        (b, e) = osp.splitext(base)
        # Key may be the same as the prefix, which will have to be at east 3
        # characters.
        m = re.match("HVC(?P<key>\d*(?P<prefix>\d{3}))", b)
        # TODO: Fall-back proceedured for when video file doesn't match HVC* pat
        return m.groupdict()['prefix'], m.groupdict()['key']

    @staticmethod
    def tempify_filename(file_path):
        """
        Generic way of generating a paired temporary file name for the given
        file name/path. Literally just adds ".TEMP" to the end of the file
        name/path.

        :param file_path: The file name or path to "tempify"
        :type file_path: str
        :return: New file path with an added suffix to it.
        :rtype: str

        """
        return file_path + ".TEMP"

    @staticmethod
    def create_dir(dir_path):
        """
        Create a directory if it doesn't already exist.

        :raises OSError: If directory doesn't already exist and couldn't be
            created.

        :param dir_path: Directory path to create if it doesn't exist.
        :type dir_path: str
        :return: The given directory path in absolute form.
        :rtype: str

        """
        log = logging.getLogger(__name__)
        dir_path = osp.abspath(osp.expanduser(dir_path))
        try:
            os.makedirs(dir_path)
            log.debug("Created directory: %s", dir_path)
        except OSError, ex:
            if ex.errno == 17 and osp.isdir(dir_path):
                log.debug("Directory already exists (%s)", dir_path)
            else:
                raise
        return dir_path

    def mp4_video_properties(self, video_file):
        """
        Get video width, height, duration and fps properties, returning a tuple
        of values corresponding to that order. RuntimeErrors are raised when
        things go bad.

        :param video_file: The path to the video file to inspect.
        :type video_file: str
        :return: Video width, height, duration (seconds) and FPS (frames per
            second).
        :rtype: (int, int, float, float)

        """
        ffprobe_exe = self.config.get(self.PROBE_CONFIG_SECT, 'ffprobe_exe')
        use_tbr_fallback = self.config.getboolean(self.PROBE_CONFIG_SECT,
                                                  'use_TBR_fallback')
        self._log.debug("using ffprobe: %s", ffprobe_exe)

        cmd = [ffprobe_exe, '-i', video_file]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        # searchable output is in err stream

        if p.returncode:  # non-0
            raise RuntimeError("ffprobe execution returned with %d code. err:\n"
                               "%s" % (p.returncode, err))

        float_match = "[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?"

        # WxH
        m = re.search("Stream.*Video.* (\d+)x(\d+)", err)
        if m:
            width = int(m.group(1))
            height = int(m.group(2))
        else:
            self._log.error("Couldn't find width/height specification for "
                            "video file '%s'", video_file)
            raise RuntimeError("Couldn't find width/height specification "
                               "for video file '%s'" % video_file)

        # FPS
        m = re.search("Stream.*Video.* (%s) fps" % float_match, err)
        if m:
            fps = float(m.group(1))
        elif use_tbr_fallback:
            # falling back on tbr measurement
            self._log.debug("Couldn't find fps measurement, looking for TBR")
            m = re.search("Stream.*Video.* (%s) tbr" % float_match, err)
            if m:
                fps = float(m.group(1))
            else:
                self._log.error("Couldn't find tbr specification for "
                                "video file '%s'", video_file)
                raise RuntimeError("Couldn't find tbr specification for "
                                   "video file '%s'" % video_file)
        else:
            self._log.error("Couldn't find fps specification for video file "
                            "'%s'", video_file)
            raise RuntimeError("Couldn't find fps specification for video file "
                               "'%s'" % video_file)

        # Duration
        m = re.search("Duration: (\d+):(\d+):(%s)" % float_match, err)
        if m:
            duration = (
                (60 * 60 * int(m.group(1)))     # hours
                + (60 * int(m.group(2)))        # minutes
                + float(m.group(3))             # seconds
            )
        else:
            raise RuntimeError("Couldn't find video duration specification")

        video_bname = osp.basename(video_file)
        self._log.debug("[%s] width           : %d", video_bname, width)
        self._log.debug("[%s] height          : %d", video_bname, height)
        self._log.debug("[%s] duration        : %f", video_bname, duration)
        self._log.debug("[%s] fps             : %f", video_bname, fps)
        self._log.debug("[%s] predicted frames: %d", video_bname,
                        int(round(duration * fps)))

        return width, height, duration, fps

    def _exclusive_touch(self, file_path):
        """
        Attempt to touch a file. If that file already exists, we return False.
        If the file was touched and created, we return True. Other OSErrors
        thrown beside the expected "file already exists" error are passed
        upwards.

        :param file_path: Path to the file to touch.
        :type file_path: str

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

    def mp4_extract_video_frames(self, video_file, output_image_ext='png'):
        """
        Extract frames out of an MP4 video file using the configured
        frame_extractor utility.

        This method will extract imagery into the configured image root
        directory with the following pattern:

            .../<image_root_dir>
                |-- <video_prefix>
                |   |-- <video_key>
                |   |   |-- <decimal_frame_time>-<frame_number>.png
                |   |   |-- ...
                |   |-- ...
                |-- ...
                ...

        where video prefix and key are as extracted from the
        ``get_video_prefix`` method from this class. Frame numbers are 6-digit
        0-padded integers.

        We return a dictionary mapping extracted video frame number to the path
        of the frame image file. We also return a dictionary mapping frame
        numbers to the frame's time relative to the start of the video. Frame
        index's in the returned maps are 0-indexed.

        If frames have already been extracted for this video in the given root
        output directory, this function does nothing except locate files and
        return declared data structures.

        :raises RuntimeError: Failed to extract any frames, i.e. out right
            extraction error, or no frames predicted or produced.

        :param video_file: The video file to extract frames from.
        :type video_file: str
        :param output_image_ext: The desired output image extension type. This
            is 'png' by default.
        :type output_image_ext: str
        :return: Two dictionaries, one mapping frame number to file path, and
            second mapping frame number to the frame time relative to the start
            of the video.
        :rtype: (dict, dict)

        """
        frm_extract_exe = self.config.get(self.EXTRACTOR_CONFIG_SECT,
                                          'frame_extractor_exe')
        second_offset = self.config.get(self.EXTRACTOR_CONFIG_SECT,
                                        'start_offset_seconds')
        second_interval = self.config.get(self.EXTRACTOR_CONFIG_SECT,
                                          'second_interval')
        max_duration_minutes = self.config.get(self.EXTRACTOR_CONFIG_SECT,
                                               'max_duration_minutes')

        # deal with configuration defaults when not provided.
        if not second_offset:
            second_offset = 0
        else:
            second_offset = float(second_offset)

        if not second_interval:
            second_interval = 0
        else:
            second_interval = float(second_interval)

        if not max_duration_minutes:
            max_duration_minutes = 0
        else:
            max_duration_minutes = float(max_duration_minutes)

        # Other variables
        file_prefix, file_key = self.get_video_prefix(video_file)
        w, h, duration, fps = self.mp4_video_properties(video_file)
        video_frame_dir = osp.join(self.image_root_dir, file_prefix, file_key)
        self._log.info("video extraction directory: %s", video_frame_dir)

        def file_name(index):
            """
            Generated standard output file path. This produces full paths, not
            just file names.
            """
            return osp.join(
                video_frame_dir,
                "HVC%s-%010.3f-%06d.%s"
                % (file_key, index / fps, index, output_image_ext)
            )

        # Predict the files that should exist with the configured settings.
        # Check some or all exist, producing those that do not.
        self._log.info("Predicting frame numbers for current settings")
        self._log.debug(" %f offset :: %f interval :: %f max duration",
                        second_offset, second_interval, max_duration_minutes)
        num_frames = int(round(duration * fps))
        frame_indicies = \
            self._get_frames_for_interval(num_frames, fps, second_offset,
                                          second_interval,
                                          max_duration_minutes * 60)

        # If no frame indices predicted to be extracted based on video
        # metadata,
        if not frame_indicies:
            raise RuntimeError("No predicted frames based on configuration!")

        # Return data structures
        frame_to_file = dict((i, file_name(i)) for i in frame_indicies)
        frame_to_time = dict((i, i / fps) for i in frame_indicies)

        self._log.debug("predicted frames for config: %s", frame_to_file)

        ###
        # Check to see if another process is working on this video via a lock
        # file. If lock file present, wait until other process has completed
        # before determining what we need to produce, as the other process may
        # be trying to extract the same frames as us (don't want to duplicate
        # work).
        #
        lock_dir = osp.join(self.image_root_dir, 'locks')
        self.create_dir(lock_dir)
        self._video_lock_file = osp.join(lock_dir, '%s.lock' % file_key)
        self._log.info("Acquiring file lock...")
        while not self._exclusive_touch(self._video_lock_file):
            time.sleep(0.01)
        self._log.info("--> Lock acquired!")

        # Now with the lock file there, try to do some stuff, protected with a
        # ``finally`` clause that removes the lock file regardless of what
        # happens.
        try:
            ###
            # Check what files exist and what ones need processing.
            #
            frames_to_process = []
            for i, img_file in sorted(frame_to_file.items()):
                # If file doesn't exist, add it to the frame-to-file map,
                # indicating that it needs processing.
                if not osp.isfile(img_file):
                    self._log.debug('frame %d needs processing', i)
                    frames_to_process.append(i)

            ###
            # Perform processing on still needed files.
            #
            if frames_to_process:
                self._log.info("Generating the following not found frame "
                               "indices: %s", frames_to_process)

                tmp_extraction_dir = video_frame_dir + ".TEMP"
                if osp.isdir(tmp_extraction_dir):
                    self._log.info("Existing TEMP folder found. Removing and "
                                   "starting over. (don't know where it left "
                                   "off last time)")
                    shutil.rmtree(tmp_extraction_dir, ignore_errors=True)
                self.create_dir(tmp_extraction_dir)

                # Converting into frame specification for extraction tool
                str_frame_selection = ','.join(map(str, frames_to_process))

                cmd = (frm_extract_exe,
                       '-i', video_file,
                       '-f', str_frame_selection,
                       '-o', osp.join(tmp_extraction_dir,
                                      "%%06d.%s" % output_image_ext))
                self._log.debug("command: '%s'", ' '.join(cmd))

                retc = subprocess.call(cmd)
                if retc != 0:
                    raise RuntimeError("Failed to run frame_extractor tool.")
                else:
                    # make primary containing dir if it doesn't exist yet.
                    self.create_dir(video_frame_dir)

                    generated_files = [osp.join(tmp_extraction_dir, f)
                                       for f in
                                       sorted(os.listdir(tmp_extraction_dir))]

                    # there better be just as many frames extracted as we has
                    # asked there to be...
                    if not len(generated_files) == len(frames_to_process):
                        self._log.warning(
                            "Mis-matched amount of frames produced vs. frames "
                            "requested.\n"
                            "-> Requested:\n"
                            "%s\n"
                            "-> Produced:\n"
                            "%s",
                            frames_to_process,
                            [osp.basename(f) for f in generated_files]
                        )

                        # figure out what files didn't get generated and
                        # remove not-produced frames from frame_to_file and
                        # frame_to_time maps.
                        actual_frames = set([
                            int(osp.splitext(osp.basename(f))[0])
                            for f in generated_files
                        ])
                        self._log.debug("actual frames produced: %s",
                                        actual_frames)

                        # Only compare between frames that should have been
                        # processed and those actually found, not the entire set
                        # of frames for this video.
                        for i in frames_to_process:
                            if i not in actual_frames:
                                self._log.warning('Frame [%d] not actually '
                                                  'produced this round, '
                                                  'removing from predicted '
                                                  'mappings', i)
                                # remove the not-produced frame from the
                                # prediction mappings
                                del frame_to_file[i]
                                del frame_to_time[i]

                    # If all frames have been removed because of mal-production,
                    # we clearly failed...
                    if not frame_to_file:
                        raise RuntimeError('Failed to extract any frames for '
                                           'video!')

                    # Run through extracted frames, renaming them into the final
                    # output directory, removing temp directory when done.
                    if generated_files:
                        self._log.info("Renaming files to have timestamp "
                                       "metadata in the filename.")
                    for index, gf in zip(frames_to_process, generated_files):
                        os.rename(gf, frame_to_file[index])
                    os.removedirs(tmp_extraction_dir)

                    self._log.info("Frame extraction complete")

            else:
                self._log.info("Frames already extracted. Skipping.")

            return frame_to_file, frame_to_time

        finally:
            # Removing lock file regardless of outcome
            os.remove(self._video_lock_file)

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

        """
        frames_taken = []

        # run through frames, counting the seconds per frame. when time interval
        # elapses, take the last frame encountered.
        cur_time = 0.0
        frame_time_interval = 1.0 / fps
        next_threshold = None

        self._log.debug("Starting frame gathering...")
        for frame in xrange(num_frames):
            self._log.debug("Frame >> %d", frame)
            self._log.debug("... Cur frame time: %.16f -> %.16f",
                            cur_time, round(cur_time, 3))

            # If we has surpassed our given maximum duration, kick out
            if max_duration and cur_time >= max_duration:
                break

            if round(cur_time, 3) >= round(offset, 3):
                if frames_taken:
                    if round(cur_time, 3) >= round(next_threshold, 3):
                        self._log.debug("... T exceeded, gathering frame")
                        # take frame, set next threshold
                        frames_taken.append(frame)
                        next_threshold += interval
                else:
                    self._log.debug("... First frame")
                    # first valid frame seen, this is our starting frame
                    frames_taken.append(frame)
                    next_threshold = cur_time + interval

                self._log.debug("... Next T: %.16f -> %.16f",
                                next_threshold, round(next_threshold, 3))

            cur_time += frame_time_interval

        return frames_taken
