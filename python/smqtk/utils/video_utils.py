import logging
import hashlib
import multiprocessing
import os
import re
import shutil
import subprocess
import time
import six

from smqtk.utils import file_utils, string_utils


__author__ = "paul.tunison@kitware.com"


class VideoMetadata (object):
    """
    Simple container for video file metadata values
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


def get_metadata_info(video_filepath, ffprobe_exe='ffprobe'):
    """
    Use ffmpeg to extract video file metadata parameters

    :param video_filepath: File path to the video to probe.
    :type video_filepath: str

    :param ffprobe_exe: Path to the ffprobe executable to use. By default, we
        try to use the version that's on the PATH.

    :return: VideoMetadata instance
    :rtype: VideoMetadata

    """
    log = logging.getLogger('smqtk.utils.video_utils.get_metadata_info')
    re_float_match = "[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?"

    log.debug("Using ffprobe: %s", ffprobe_exe)
    cmd = [ffprobe_exe, '-i', video_filepath]
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
                           "for video file '%s'" % video_filepath)

    # FPS
    m = re.search("Stream.*Video.* (%s) fps" % re_float_match, err)
    if m:
        fps = float(m.group(1))
    else:
        # falling back on tbr measurement
        log.debug("Couldn't find fps measurement, looking for TBR")
        m = re.search("Stream.*Video.* (%s) tbr" % re_float_match, err)
        if m:
            fps = float(m.group(1))
        else:
            raise RuntimeError("Couldn't find tbr specification for "
                               "video file '%s'" % video_filepath)

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
                           "video file '%s'" % video_filepath)

    md = VideoMetadata()
    md.width = width
    md.height = height
    md.fps = fps
    md.duration = duration

    return md


def ffmpeg_extract_frame(t, input_filepath, output_filepath,
                         ffmpeg_exe='ffmpeg'):
    """
    Extract a frame a the given time ``t`` from the input video file.
    Output file may not exist or be of 0 size if we failed to extract the frame.
    """
    cmd = [ffmpeg_exe, '-accurate_seek', '-ss', str(t), '-i', input_filepath,
           '-frames:v', '1', output_filepath]
    sPIPE = subprocess.PIPE
    p = subprocess.Popen(cmd, stdout=sPIPE, stderr=sPIPE)
    _, _ = p.communicate()
    # if p.returncode != 0:
    #     raise RuntimeError("FFmpeg failed to extract frame at time %f! "
    #                        "(return code: %d)"
    #                        % (t, p.returncode))


def ffmpeg_extract_frame_map(working_dir, video_filepath, second_offset=0,
                             second_interval=0, max_duration=0, frames=(),
                             output_image_ext="png", parallel=None,
                             ffmpeg_exe='ffmpeg'):
    """
    Return a mapping of video frame index to image file in the given output
    format.

    If frames requested have not yet been extracted (based on what's contained
    in the specified output directory), they are done now. This means that this
    method could take a little time to complete if there are many frames in the
    video file and this is the first time this is being called.

    This may return an empty list if there are no frames in the video for
    the specified, or default, constraints.

    Extracted frames are cached in a directory structure under the provided
    ``working_dir`` directory path: ``<working_dir>/VideoFrameExtraction``.
    Frames are extracted into separate directories based on the SHA1 checksum of
    the video file.

    :raises RuntimeError: No frames were extracted.

    :param working_dir: Working directory for frame extraction to occur in.
    :type working_dir: str

    :param video_filepath: Path to the video to extract frames from.
    :type video_filepath: str

    :param second_offset: Seconds into the video to start extracting
    :type second_offset: float

    :param second_interval: Number of seconds between extracted frames
    :type second_interval: float

    :param max_duration: Maximum number of seconds worth of extracted frames
        (starting from the specified offset). If <=0, we extract until the end
        of the video.
    :type max_duration: float

    :param frames: Specific exact frame numbers within the video to extract.
        Providing explicit frames causes offset, interval and duration
        parameters to be ignored and only the frames specified here to be
        extracted and returned.
    :type frames: collections.Iterable[int]

    :param parallel: Number of processes to use for frame extraction. This is
        None by default, meaning that all available cores/threads are used.
    :type parallel: int or None

    :param ffmpeg_exe: ffmpeg executable to use for frame extraction. By
        default, we attempt to use what is available of the PATH.
    :type ffmpeg_exe: str or unicode

    :return: Map of frame-to-filepath for requested video frames
    :rtype: dict of (int, str)

    """
    log = logging.getLogger('smqtk.utils.video_utils.extract_frame_map')

    video_md = get_metadata_info(video_filepath)
    video_sha1sum = hashlib.sha1(open(video_filepath, 'rb').read()).hexdigest()
    frame_output_dir = os.path.join(
        working_dir,
        "VideoFrameExtraction",
        *string_utils.partition_string(video_sha1sum, 10)
        # 40 hex chars split into chunks of 4
    )
    file_utils.safe_create_dir(frame_output_dir)

    def filename_for_frame(frame, ext):
        """
        method standard filename for a given frame file
        """
        return "%08d.%s" % (frame, ext.lstrip('.'))

    def iter_frames_for_interval():
        """
        Return a generator expression yielding frame numbers from the input
        video that match the given query parameters. Indices returned are
        0-based (i.e. first frame is 0, not 1).

        We are making a sensible assumption that we are not dealing with frame
        speeds of over 1000Hz and rounding frame frame times to the neared
        thousandth of a second to mitigate floating point error.

        :rtype: list of int

        """
        _log = logging.getLogger('smqtk.utils.video_utils.extract_frame_map'
                                 '.iter_frames_for_interval')
        num_frames = int(video_md.fps * video_md.duration)
        first_frame = second_offset * video_md.fps
        _log.debug("First frame: %f", first_frame)
        if max_duration > 0:
            cutoff_frame = min(float(num_frames),
                               (max_duration + second_offset) * video_md.fps)
        else:
            cutoff_frame = float(num_frames)
        _log.debug("Cutoff frame: %f", cutoff_frame)
        if second_interval:
            incr = second_interval * video_md.fps
        else:
            incr = 1.0
        _log.debug("Frame increment: %f", incr)

        # Interpolate
        yield first_frame
        next_frm = first_frame + incr
        while next_frm < cutoff_frame:
            _log.debug("-- adding frame: %f", next_frm)
            yield int(next_frm)
            next_frm += incr

    def extract_frames(frames_to_process):
        """
        Extract specific frames from the input video file using ffmpeg. If not
        all frames could be extracted, we return what we were able to extract.

        :param frames_to_process: Mapping of frame-number:filepath pairs to
            extract from the input video.
        :type frames_to_process: dict[int,str or unicode]

        :return: List of frames that were successfully extracted.
        :rtype: list[int]

        """
        _log = logging.getLogger('smqtk.utils.video_utils.extract_frame_map'
                                 '.extract_frames')

        # Setup temp extraction directory
        tmp_extraction_dir = os.path.join(frame_output_dir, ".TMP")
        if os.path.isdir(tmp_extraction_dir):
            _log.debug("Existing temp director found, removing and starting "
                       "over")
            shutil.rmtree(tmp_extraction_dir, ignore_errors=True)
        os.makedirs(tmp_extraction_dir)

        p = multiprocessing.Pool(parallel)
        # Mapping of frame to (result, output_filepath)
        #: :type: dict of (int, (AsyncResult, str))
        rmap = {}
        for f, ofp in six.iteritems(frames_to_process):
            tfp = os.path.join(tmp_extraction_dir,
                               filename_for_frame(f, output_image_ext))
            t = f / video_md.fps
            rmap[f] = (
                p.apply_async(ffmpeg_extract_frame,
                              args=(t, video_filepath, tfp, ffmpeg_exe)),
                tfp
            )
        p.close()
        # Check for failures
        extracted_frames = []
        for f, ofp in six.iteritems(frames_to_process):
            r, tfp = rmap[f]
            r.get()  # wait for finish
            if not os.path.isfile(tfp):
                _log.warn("Failed to generated file for frame %d", f)
            else:
                extracted_frames.append(f)
                os.rename(tfp, ofp)
        p.join()
        del p

        os.removedirs(tmp_extraction_dir)
        _log.debug("Frame extraction complete")

        return extracted_frames

    # Determine frames to extract from video
    extract_indices = set()
    if frames:
        log.debug("Only extracting specified frames: %s", frames)
        extract_indices.update(frames)
    else:
        log.debug("Determining frames needed for specification: "
                  "offset: %f, interval: %f, max_duration: %f",
                  second_offset, second_interval, max_duration)
        extract_indices.update(iter_frames_for_interval())

    if not extract_indices:
        return {}

    # frame/filename map that will be returned based on requested frames
    frame_map = dict(
        (i, os.path.join(frame_output_dir,
                         filename_for_frame(i, output_image_ext)))
        for i in extract_indices
    )

    ###
    # Acquire a file-base lock in output directory so that we don't conflict
    # with another process extracting frames to the same directory.
    #
    # NOTE: This method is prone to starvation if many processes are trying
    #       to extract to the same video frames, but not yet probably due to
    #       existing use cases.
    #
    lock_file = os.path.join(frame_output_dir, '.lock')
    log.debug("Acquiring file lock in '%s'...", frame_output_dir)
    while not file_utils.exclusive_touch(lock_file):
        # This is sufficiently small to be fine grained, but not so small to
        # burn the CPU.
        time.sleep(0.01)
    log.debug("Acquiring file lock -> Acquired!")

    try:
        ###
        # Determine frames to actually extract base on existing files (if any)
        #
        #: :type: dict[int, str]
        frames_to_process = {}
        existing_frames = []
        for i, img_file in sorted(frame_map.items()):
            if not os.path.isfile(img_file):
                log.debug('frame %d needs processing', i)
                frames_to_process[i] = img_file
            else:
                existing_frames.append(i)

        ###
        # Extract needed frames via hook function that provides
        # implementation.
        #
        if frames_to_process:
            frames_extracted = extract_frames(frames_to_process)

            if (len(existing_frames) + len(frames_extracted)) == 0:
                raise RuntimeError("Failed to extract any frames for video")

        return frame_map
    finally:
        os.remove(lock_file)
