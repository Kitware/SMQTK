"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import abc
import logging
import multiprocessing.pool
import numpy
import os
import os.path as osp
# noinspection PyPackageRequirements
import PIL.Image
import subprocess
import tempfile

from smqtk.content_description import ContentDescriptor

from .utils import DescriptorIO
from . import encode_FLANN


def _async_cd_process_helper(cd_util, detector_type, work_directory,
                             image_filepath, frame, ds_spacing):
        """
        Worker method for generating descriptor matrix via the colorDescriptor
        tool.

        :param cd_util: colorDescriptor utility executable to use (path).
        :type cd_util: str

        :param detector_type: ColorDescriptor detector type to use
        :type detector_type: str

        :param work_directory: Work directory to place temporary files
        :type work_directory: str

        :param image_filepath: Image file to generate a descriptor matrix for
        :type image_filepath: str

        :return: Descriptor matrix
        :rtype: numpy.matrix

        """
        log = logging.getLogger("ColorDescriptor_Base._async_cd_process_helper")
        log.debug("Async work{ cd_util: %s, detector_type: %s, "
                  "work_directory: %s, image_filepath: %s, ds_spacing: %s }",
                  cd_util, detector_type, work_directory, image_filepath,
                  ds_spacing)

        tmp_fd, tmp_file = tempfile.mkstemp(dir=work_directory)

        def tmp_clean():
            os.remove(tmp_file)
            os.close(tmp_fd)

        cmd = [cd_util, image_filepath,
               '--detector', 'densesampling',
               '--ds_spacing', str(ds_spacing),
               '--descriptor', detector_type,
               '--output', tmp_file]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            tmp_clean()
            raise RuntimeError("Failed to fun colorDescriptor executable for "
                               "file \"%(file)s\" (command: %(cmd)s)\n"
                               "Output:\n%(out)s\n"
                               "Error :\n%(err)s"
                               % {"file": image_filepath,
                                  "cmd": cmd,
                                  "out": out,
                                  "err": err})

        # Read in descriptor output from file and convert to matrix form
        info, descriptors = DescriptorIO.readDescriptors(tmp_file)
        tmp_clean()

        # number of descriptor elements in this image
        n = info.shape[0]
        return numpy.hstack((
            numpy.zeros((n, 1)) + frame,
            info[:, 0:2],
            descriptors
        ))


# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_Base (ContentDescriptor):
    """
    CSIFT colordescriptor feature descriptor

    This generates histogram-based features.

    """

    # colordescriptor executable that should be on the PATH
    PROC_COLORDESCRIPTOR = 'colorDescriptor'

    # Class detector type string to pass to colorDescriptor exe
    DETECTOR_TYPE = None  # e.g. "csift" or "transformedcolorhistogram"

    def __init__(self, data_directory, work_directory):
        super(ColorDescriptor_Base, self).__init__(data_directory,
                                                   work_directory)
        # Expected codebook filepath to use for quantization
        self._codebook_file = osp.join(self.data_directory, "codebook.npy")

    @abc.abstractmethod
    def _get_flan_file_components(self):
        """
        :return: Paths to FLANN component files in (codebook, flann-file) format
        :rtype: (str, str)
        """
        pass

    @abc.abstractmethod
    def _get_data_width_height(self, data):
        """
        :return: Get the pixel width and height of the given data element.
        :rtype: (int, int)
        """
        pass

    @abc.abstractmethod
    def _generate_descriptor_matrix(self, data, ds_spacing):
        """
        Generate the descriptor matrix for the given data file.

        :param data: Data file to base descriptor generation on
        :type data: DataFile

        :return: Descriptor matrix
        :rtype: numpy.matrix

        """
        pass

    def compute_feature(self, data):
        """
        Compute CSIFT colordescriptor feature given a VideoFile instance.

        :param data: Video file wrapper
        :type data:
            smqtk.utils.DataFile.DataFile or smqtk.utils.VideoFile.VideoFile

        :return: Video feature vector
        :rtype: numpy.ndarray

        """
        # Check for checkpoint file, if exists, just return the loaded feature
        # vector
        # check_point_file = osp.join(osp.join(self.work_directory,
        #                                      *data.split_md5sum(8)[:-1]),
        #                             "%s.feature.npy" % data.md5sum)
        # if osp.isfile(check_point_file):
        #     return numpy.load(check_point_file)

        self.log.debug("Processing video: %s", data.filepath)

        ###
        # For pixel sample grid, we want to take at a maximum of 50,
        # sample points in longest direction with at least a 6 pixel spacing. We
        # will take fewer sample points to ensure the 6 pixel minimum spacing.
        # (magic numbers are a result of tuning, see Sangmin)
        #
        # Using min instead of max due to images that are long and thin, and
        # vice versa, which, when using max, would cause some quadrants to have
        # no detections (see spHist below)
        #
        # ds_spacing = max(int(max(w, h) / 50.0), 6)
        w, h = self._get_data_width_height(data)
        ds_spacing = max(int(min(w, h) / 50.0), 6)
        self.log.debug("Calculated ds_spacing: %f", ds_spacing)

        ###
        # Create descriptor matrix from colorDescriptor tool output
        #
        # - This ends up boiling down to one or more calls to
        #   _async_cd_process_helper.
        #
        descriptor_matrix = self._generate_descriptor_matrix(data, ds_spacing)

        ###
        # Encode/Quantize result descriptor matrix
        #
        flann_codebook, flann_file = self._get_flan_file_components()
        quantized = encode_FLANN.quantizeResults3(descriptor_matrix,
                                                  flann_codebook, flann_file)
        self.log.debug("quantized :: %s\n%s", quantized.shape, quantized)

        ###
        # Create spacial pyramid histogram
        #
        # Result of build_sp_hist is an 8xN matrix, where each row is a
        # clip-level feature for a spacial region. Final feature product
        # will composed of 4 of the 8 vectors (full image + image thirds)
        #
        sp_hist = encode_FLANN.build_sp_hist2(quantized)
        self.log.debug("sphist :: %s\n%s",
                       sp_hist.shape, sp_hist)
        self.log.debug("sphist sums: \n%s", sp_hist.sum(axis=1))

        ###
        # Combine SP histogram into single vector and return
        #
        # normalizing each "quadrant" so their sum is a quarter of the feature
        # total (this 4x multiplier on each vector norm)
        #
        q1 = sp_hist[0] / (sp_hist[0].sum()*4.0)
        q2 = sp_hist[5] / (sp_hist[5].sum()*4.0)
        q3 = sp_hist[6] / (sp_hist[6].sum()*4.0)
        q4 = sp_hist[7] / (sp_hist[7].sum()*4.0)

        feature = numpy.hstack((q1, q2, q3, q4))

        # Diagnostic introspection logging
        # self.log.debug("feature :: %s shape=%s dtype=%s",
        #                type(feature), feature.shape, feature.dtype)
        # self.log.debug("\tnormalized max: %s", feature.max())
        # self.log.debug("\tnormalized sum: %s", feature.sum())
        # self.log.debug("\tnormalized sum 1/4: %s", feature[:4096].sum())
        # self.log.debug("\tnormalized sum 2/4: %s",
        #                feature[4096*1:4096*2].sum())
        # self.log.debug("\tnormalized sum 3/4: %s",
        #                feature[4096*2:4096*3].sum())
        # self.log.debug("\tnormalized sum 4/4: %s",
        #                feature[4096*3:4096*4].sum())

        # # write check-point file with feature vector
        # self.log.debug("Saving feature vector checkpoint: %s", check_point_file)
        # safe_create_dir(osp.dirname(check_point_file))
        # numpy.save(check_point_file, feature)

        return feature


# noinspection PyAbstractClass
class ColorDescriptor_Image (ColorDescriptor_Base):
    """
    Image-based implementation of abstract functions
    """

    def _get_data_width_height(self, data):
        """
        :param data: DataFile to get the properties from
        :type data: SMQTK.utils.DataFile.DataFile

        :return: Get the pixel width and height of the given data element in
            (width, height) format..
        :rtype: (int, int)

        """
        return PIL.Image.open(data.filepath).size

    def _generate_descriptor_matrix(self, data, ds_spacing):
        """
        Generate the descriptor matrix for the given data file.

        :param data: Data file to base descriptor generation on
        :type data: SMQTK.utils.DataFile.DataFile

        :return: Descriptor matrix
        :rtype: numpy.matrix

        """
        # one image, so single call to generator helper
        return _async_cd_process_helper(self.PROC_COLORDESCRIPTOR,
                                        self.DETECTOR_TYPE,
                                        self.work_directory, data.filepath,
                                        0, ds_spacing)


# noinspection PyAbstractClass
class ColorDescriptor_Video (ColorDescriptor_Base):
    """
    Video-based implementation of abstract functions
    """

    def _get_data_width_height(self, data):
        """
        :param data: DataFile to get the properties from
        :type data: SMQTK.utils.VideoFile.VideoFile

        :return: Get the pixel width and height of the given data element in
            (width, height) format..
        :rtype: (int, int)

        """
        md = data.metadata()
        return md.width, md.height

    def _generate_descriptor_matrix(self, data, ds_spacing):
        """
        Generate the descriptor matrix for the given data file.

        :param data: Data file to base descriptor generation on
        :type data: SMQTK.utils.VideoFile.VideoFile

        :return: Descriptor matrix
        :rtype: numpy.matrix

        """
        # Creating subdirectory to put video-specific work files in
        # video_work_dir = osp.join(self.work_directory,
        #                           *data.split_md5sum(8))
        # if not osp.isdir(video_work_dir):
        #     os.makedirs(video_work_dir)

        # Extract frame from the video to process over
        # - Cover every 2 seconds between 20% -> 80% time points of video
        self.log.debug("[%s] getting video frames", data)
        frame_map = data.frame_map(data.metadata().duration * 0.2, 2,
                                   data.metadata().duration * 0.6)
        ordered_frame_list = sorted(frame_map.keys())

        self.log.debug("[%s] Submitting colorDescriptor jobs", data)
        # Using a thread pool since underlying ta[sks are primarily within a
        # subprocess and outside the GIL
        p = multiprocessing.Pool(processes=self.PARALLEL)
        #: :type: dict of (int, multiprocessing.pool.ApplyResult)
        result_map = {}
        for frame in ordered_frame_list:
            result_map[frame] = \
                p.apply_async(_async_cd_process_helper,
                              args=(self.PROC_COLORDESCRIPTOR,
                                    self.DETECTOR_TYPE, self.work_directory,
                                    frame_map[frame], frame, ds_spacing))

        self.log.debug("[%s] Combining colorDescriptor results", data)
        combined_matrix = None
        for frame in ordered_frame_list:
            frame_d_mat = result_map[frame].get()
            if combined_matrix is None:
                combined_matrix = frame_d_mat
            else:
                combined_matrix = numpy.vstack((combined_matrix, frame_d_mat))

        p.close()
        p.join()

        return combined_matrix


# noinspection PyAbstractClass
class ColorDescriptor_CSIFT (ColorDescriptor_Base):
    DETECTOR_TYPE = "csift"

    def _get_flan_file_components(self):
        """
        :return: Paths to FLANN component files in (codebook, flann-file) format
        :rtype: (str, str)
        """
        return (osp.join(osp.dirname(__file__), "csift_codebook_med12.txt"),
                osp.join(osp.dirname(__file__), "csift.flann"))


# noinspection PyAbstractClass
class ColorDescriptor_TCH (ColorDescriptor_Base):
    DETECTOR_TYPE = "transformedcolorhistogram"

    def _get_flan_file_components(self):
        """
        :return: Paths to FLANN component files in (codebook, flann-file) format
        :rtype: (str, str)
        """
        return (osp.join(osp.dirname(__file__), "tch_codebook_med12.txt"),
                osp.join(osp.dirname(__file__), "tch.flann"))


# ColorDescriptor descriptor combinations
# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_CSIFT_Image (ColorDescriptor_CSIFT,
                                   ColorDescriptor_Image):
    """ CSIFT descriptor over image files """


# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_CSIFT_Video (ColorDescriptor_CSIFT,
                                   ColorDescriptor_Video):
    """ CSIFT descriptor over video files """


# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_TCH_Image (ColorDescriptor_TCH,
                                 ColorDescriptor_Image):
    """ TCH descriptor over image files """


# noinspection PyAbstractClass,PyPep8Naming
class ColorDescriptor_TCH_Video (ColorDescriptor_TCH,
                                 ColorDescriptor_Video):
    """ TCH descriptor over video files """
