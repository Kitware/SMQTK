"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import math
import multiprocessing
import numpy
import os
import os.path as osp
import shutil
import subprocess

from SMQTK.FeatureDescriptor import FeatureDescriptor

from . import DescriptorIO
from . import encode_FLANN


class ColorDescriptor_CSIFT_Video (FeatureDescriptor):
    """
    CSIFT colordescriptor feature descriptor
    """

    # colordescriptor executable that should be on the PATH
    PROC_COLORDESCRIPTOR = 'colorDescriptor'

    def __init__(self, base_data_directory, base_work_directory):
        super(ColorDescriptor_CSIFT_Video, self).__init__(base_data_directory,
                                                          base_work_directory)

        # Required files for FLANN quantization
        self._flann_codebook = osp.join(osp.dirname(__file__),
                                        "csift_codebook_med12.txt")
        self._flann_file = osp.join(osp.dirname(__file__),
                                    "csift.flann")

        # Standard filenames for stage components
        self._combined_filename = "csift.combined.txt"
        self._quantized_filename = "csift.quantized.txt"
        self._sp_hist_filename = "csift.sp_hist.txt"

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

        ###
        # Create combined feature file from per-frame features
        # Only extracting every 2 seconds to get sparse representation.
        # Skipping if file already present.
        #
        work_combined_features = osp.join(video_work_dir,
                                          self._combined_filename)
        if not osp.isfile(work_combined_features):
            frame_map = data.frame_map(0, 2)
            self._generate_combined_features(data, frame_map, video_work_dir,
                                             work_combined_features)

        ###
        # use FLANN to encode/quantize combined features
        #
        csift_feature = self._encode_FLANN(work_combined_features,
                                           video_work_dir)

        return csift_feature

    def _generate_combined_features(self, video_data, frame_map, video_work_dir,
                                    combined_features_output):
        """
        Generate CSIFT combined features to file given a set of video frames.

        This method generates N intermediary per-frame output files from the
        colordescriptor executable in order to combine them into the final
        output file. Intermediate per-frame files are removed once the combined
        file is successfully generated.

        :param video_data: Video data object we're processing over
        :type video_data: SMQTK.utils.VideoFile.VideoFile

        :param frame_map: Video frame number-to-file association
        :type frame_map: dict of (int, str)

        :param video_work_dir: Base working directory for the current video.
        :type video_work_dir: str

        :param combined_features_output: Output file to save combined features
            to.
        :type combined_features_output: str

        """
        per_frame_work_dir = osp.join(video_work_dir, 'f')
        if not osp.isdir(per_frame_work_dir):
            os.makedirs(per_frame_work_dir)

        vmd = video_data.metadata()
        w = vmd.width
        h = vmd.height

        # For pixel sample grid, we want to take at a maximum of 50,
        # sample points in longest direction with at least a 6 pixel spacing. We
        # will take fewer sample points to ensure the 6 pixel minimum spacing.
        # (magic numbers are
        # a result of tuning)
        sample_size = max(int(math.floor(max(w, h) / 50.0)), 6)

        # Output files used by this method. These files will also act like stamp
        # files, detailing progress from previous runs. Files will be removed
        # from the disk when the final product of this method has been
        # completed. When this file is present before processing occurs, a total
        # processing skip occurs.
        cd_output_file_pattern = osp.join(per_frame_work_dir, "csift-%06d.txt")
        cd_log_file_pattern = osp.join(per_frame_work_dir, "csift-%06d.log")
        # Files will be added to these maps keyed by their frame/index number
        csift_frame_feature_files = {}

        def construct_cd_command(in_frame_file, out_file):
            return [self.PROC_COLORDESCRIPTOR, in_frame_file,
                    '--detector', 'densesampling',
                    '--ds_spacing', str(sample_size),
                    '--descriptor', 'csift',
                    '--output', out_file]

        # pool = multiprocessing.Pool()
        # job_results = {}
        for i, (frame, png_file) in enumerate(frame_map.items()):
            out_file = cd_output_file_pattern % frame
            if not osp.isfile(out_file):
                self.log.debug("[frame:%d] Submitting job for frame %d", frame,
                               frame)
                log_file = cd_log_file_pattern % frame
                tmp_file = out_file + '.TMP'
                if osp.isfile(tmp_file):
                    os.remove(tmp_file)

                cmd = construct_cd_command(png_file, tmp_file)
                with open(log_file, 'w') as lfile:
                    rc = subprocess.call(cmd, stdout=lfile, stderr=lfile)
                if rc != 0:
                    raise RuntimeError("Failed to process colordescriptor")
                os.rename(tmp_file, out_file)

                # cmd = construct_cd_command(png_file, tmp_file)
                # lfile = open(log_file, 'w')
                #
                # def callback(rc):
                #     if rc != 0:
                #         raise RuntimeError("Failed to process colordescriptor")
                #     self.log.debug("Finished processing for frame %d" % frame)
                #     lfile.close()
                #     os.rename(tmp_file, out_file)
                #
                # job_results[frame] = \
                #     pool.apply_async(subprocess.call, (cmd,), {'stdout': lfile,
                #                                                'stderr': lfile},
                #                      callback)
            else:
                self.log.debug("[frame:%d] CSIFT features already processed")

            csift_frame_feature_files[frame] = out_file

        # # Scan results for finish wait
        # self.log.debug("Waiting for jobs to finish")
        # for r in job_results.values():
        #     r.wait()
        # self.log.debug("-> Done!")

        self.log.debug("Combining frame features")
        tmp_combined_output = combined_features_output + ".TMP"
        self._combine_frame_features(csift_frame_feature_files,
                                     tmp_combined_output)
        os.rename(tmp_combined_output, combined_features_output)

        # With combined file completed, remove per-frame work
        self.log.debug("Cleaning up per-frame work directory...")
        def onerror(func, path, exc_info):
            self.log.warn("Error in rmtree on function [%s] -> %s\n"
                          "-- %s", str(func), path, exc_info)
        shutil.rmtree(per_frame_work_dir, onerror=onerror)

    # noinspection PyMethodMayBeStatic
    def _combine_frame_features(self, frame_feature_files,
                                output_file):
        """
        Combine descriptor output matrices into a single matrix.

        The combined data matrix representing the all given features has a
        specific format intended for use in quantization (encode_FLANN
        functions):
            [
             [ <frame_num>, <info1>, <info2>, ... <feature vector> ],
             ...
            ]

        :param frame_feature_files: Mapping of frame number to its associated
            feature file as generated from colordescriptor.
        :type frame_feature_files: dict of (int, str)

        :param output_file: The file to output the combined matrix to. If a file
            exists by this name already, it will be overwritten.
        :type output_file: str

        """
        with open(output_file, 'w') as output_file:

            for i, ff in sorted(frame_feature_files.items()):
                info, descriptors = DescriptorIO.readDescriptors(ff)

                n = info.shape[0]  # num rows
                data_frame = numpy.hstack((numpy.ones((n, 1)) * i,
                                          info[:, 0:2],
                                          descriptors))

                # Continuously adding to the same file with savetxt effectively
                # performs a v-stack operation. '%g' uses the shorter of %e or
                # %f, i.e. exponential or floating point format respectively.
                # TODO: Actually v-stack arras save to binary file (saves space)
                #       Would also require modifications to quantize step
                # TODO: Could also use mmap
                numpy.savetxt(output_file, data_frame, fmt='%g')

    def _encode_FLANN(self, combined_file, video_work_dir):
        """
        Quantize and encode the given combined matrix into the supplied
        quantized and spacial pyramid histogram files. We return a
        numpy array object of the video-level feature described by the given
        combined matrix file.

        If the quantized file and/or sphist file already exist, we load the
        existing data from those files to construct the video-level feature
        vector.

        :param combined_file: Path to the file containing the result from the
            _combine_frame_results method, which is basically matrix containing
            all frame-level feature matrices.
        :type combined_file: str

        :param video_work_dir: Base working directory for the current video
        :type video_work_dir: str

        :return: A 1D numpy array (vector) representing the video-level feature.
        :rtype: numpy.ndarray

        """
        quantized_file = osp.join(video_work_dir, self._quantized_filename)
        sphist_file = osp.join(video_work_dir, self._sp_hist_filename)

        if not osp.isfile(quantized_file):
            self.log.debug('building FLANN quantized file')
            tmp_file = quantized_file + ".TMP"
            encode_FLANN.quantizeResults2(combined_file, tmp_file,
                                          self._flann_codebook,
                                          self._flann_file)
            os.rename(tmp_file, quantized_file)
        else:
            self.log.debug('existing quantized file found')

        if not osp.isfile(sphist_file):
            self.log.debug('building FLANN spacial pyramid')
            tmp_file = sphist_file + ".TMP"
            # Returns matrix of int64 type
            sp_histogram = encode_FLANN.build_sp_hist_(quantized_file, tmp_file)
            os.rename(tmp_file, sphist_file)
        else:
            # ... So we load it back in as that type, too
            sp_histogram = numpy.loadtxt(sphist_file, dtype=numpy.int64)
            self.log.debug('existing sphist file found')

        # Histogram file will consist of 8 vectors. Unified vector is all of
        # those h-stacked.

        # Result of build_sp_hist is an 8xN matrix, where each row is a
        # clip-level feature for a spacial region. Final feature product
        # will be a 4 subset of these 8 vectors h-stacked.
        # Voila, clip level feature!
        _hist_sp = numpy.hstack(sp_histogram[[0, 5, 6, 7], :])
        hist_sp = _hist_sp / float(numpy.sum(_hist_sp))
        return hist_sp
