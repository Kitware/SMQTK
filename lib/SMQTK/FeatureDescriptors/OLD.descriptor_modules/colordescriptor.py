"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import math
import numpy as np
import os
import os.path as osp
from subprocess import call as sub_call

from ...VCDStore import VCDStoreElement
from .. import VCDWorkerInterface

from ._colordescriptor import encode_FLANN
from ._colordescriptor import DescriptorIO


class colordescriptor (VCDWorkerInterface):

    DESCRIPTOR_ID = 'colordescriptor'

    @classmethod
    def generate_config(cls, config=None):
        config = super(colordescriptor, cls).generate_config(config)

        sect = cls.DESCRIPTOR_ID
        if not config.has_section(sect):
            config.add_section(
                sect,
                "Options for colorDescriptor VCD.\n"
                "\n"
                "Creates store elements with the following descriptor IDs:\n"
                "    - csift_flann\n"
                "    - tch_flann\n"
            )

        config.set(sect, 'old_style_work_directory', '',
                   "If set, indicates that we should check this directory for "
                   "the possible existence of already computer CSIFT and TCH "
                   "combined-frame matrix files. If they are present, we skip "
                   "processing and use then for encoding, etc.")
        config.set(sect, 'colordescriptor_exe', 'colorDescriptor',
                   "Location of the colorDescriptor executable to use for "
                   "processing.")

        ### Mode specific parameters

        # FLANN
        # - default files are located in our internal _colordescriptor module
        config.set(sect, 'flann-csift_codebook',
                   osp.join(osp.dirname(__file__), '_colordescriptor',
                            'csift_codebook_med12.txt'),
                   "The location of the csift FLANN codebook file to use.")
        config.set(sect, 'flann-csift_flann_file',
                   osp.join(osp.dirname(__file__), '_colordescriptor',
                            'csift.flann'),
                   "The location of the csift FLANN file binary.")
        config.set(sect, 'flann-tch_codebook',
                   osp.join(osp.dirname(__file__), '_colordescriptor',
                            'tch_codebook_med12.txt'),
                   "The location of the tch FLANN codebook file to use.")
        config.set(sect, 'flann-tch_flann_file',
                   osp.join(osp.dirname(__file__), '_colordescriptor',
                            'tch.flann'),
                   "The location of the tch FLANN file binary.")

        return config

    def __init__(self, config, working_dir, image_root):
        super(colordescriptor, self).__init__(config, working_dir, image_root)

        self.cdescriptor_exe = self.config.get(self.DESCRIPTOR_ID,
                                               'colordescriptor_exe')

        self.old_style_work_dir = \
            self.config.has_option(self.DESCRIPTOR_ID,
                                   'old_style_work_directory') \
            and self.config.get(self.DESCRIPTOR_ID, 'old_style_work_directory')

        self.csift_flann_codebook = self.config.get(self.DESCRIPTOR_ID,
                                                    'flann-csift_codebook')
        self.csift_flann_file = self.config.get(self.DESCRIPTOR_ID,
                                                'flann-csift_flann_file')
        self.tch_flann_codebook = self.config.get(self.DESCRIPTOR_ID,
                                                  'flann-tch_codebook')
        self.tch_flann_file = self.config.get(self.DESCRIPTOR_ID,
                                              'flann-tch_flann_file')

    def process_video(self, video_file):
        """
        Process the given video file, producing video level features.

        :param video_file: The video file to process on.
        :type video_file: str

        """
        self._log.info("colordescriptor processing video: %s", video_file)

        # This descriptor has separate output files per video, so creating a
        # sub-work directory to do things in so as not to conflict with other
        # video processing.
        file_prefix, file_key = self.get_video_prefix(video_file)
        video_work_dir = osp.join(self.working_dir, self.DESCRIPTOR_ID,
                                  str(file_prefix), str(file_key))
        self.create_dir(video_work_dir)

        # Defining key file paths
        csift_combined_features_file = osp.join(video_work_dir,
                                                'csift.combined.txt')
        tch_combined_features_file = osp.join(video_work_dir,
                                              'tch.combined.txt')

        # Record space for feature store elements produced by encoding step.
        vcd_store_elements = []

        ###
        # Create Combined files from frame products
        #

        # Different code paths may compress files in different manners. The
        # active path will flip this switch if it compresses the combined files
        # via bzip2
        combined_f_is_bzipped = False

        # If we were provided an old-style working directory and an old-style
        # combined matrix file exists for both csift and tch, skip processing
        # and use these files.
        # TODO: Make this more extension dynamic (i.e. pick up gzipped or normal
        #       txt files too)
        if (self.old_style_work_dir
                and osp.isfile(osp.join(self.old_style_work_dir,
                                        '%s.csift-all.txt.bz2' % file_key))
                and osp.isfile(osp.join(self.old_style_work_dir,
                                        '%s.tcg-all.txt.bz2' % file_key))):
            combined_f_is_bzipped = True
            # Resetting combined file targets to the old-style files.
            # "old-style" as in they're not where we would usually put them.
            csift_combined_features_file = \
                osp.isfile(osp.join(self.old_style_work_dir,
                                    '%s.csift-all.txt.bz2' % file_key))
            tch_combined_features_file = \
                osp.isfile(osp.join(self.old_style_work_dir,
                                    '%s.tcg-all.txt.bz2' % file_key))

        # Skip processing if computed products already exist
        elif not (osp.isfile(csift_combined_features_file)
                  and osp.isfile(tch_combined_features_file)):
            # Extracting video frames needed. Exit with no feature elements if
            # this fails.
            try:
                frm2file, frm2time = \
                    self.mp4_extract_video_frames(video_file)
            except RuntimeError, ex:
                self._log.error("Failed frame extraction! Either process "
                                "failed or no frames were extracted! "
                                "(error: %s) Check up on video: %s",
                                str(ex), video_file)
                return None

            # TODO: Make encapsulation object for video file and accompanying
            #       data structures, so we don't have separate video file and
            #       frame-to-things maps, since they're all tied to each other.
            self._colordescriptor_generation(video_file, frm2file,
                                             video_work_dir,
                                             csift_combined_features_file,
                                             tch_combined_features_file)
        else:
            self._log.info("CSIFT/TCH combined features already computed.")

        ###
        # FLANN Encoding
        #
        csift_quantized_file = osp.join(video_work_dir, 'csift.quantized.txt')
        csift_sp_hist_file = osp.join(video_work_dir, 'csift.sp_hist.txt')
        tch_quantized_file = osp.join(video_work_dir, 'tch.quantized.txt')
        tch_sp_hist_file = osp.join(video_work_dir, 'tch.sp_hist.txt')

        csift_feature = self._encode_FLANN(
            csift_combined_features_file, csift_quantized_file,
            csift_sp_hist_file, self.csift_flann_codebook,
            self.csift_flann_file, filein_is_bzipped=combined_f_is_bzipped
        )
        vcd_store_elements.append(
            VCDStoreElement('csift_flann', int(file_key), csift_feature)
        )

        tch_feature = self._encode_FLANN(
            tch_combined_features_file, tch_quantized_file, tch_sp_hist_file,
            self.tch_flann_codebook, self.tch_flann_file,
            filein_is_bzipped=combined_f_is_bzipped
        )
        vcd_store_elements.append(
            VCDStoreElement('tch_flann', int(file_key), tch_feature)
        )

        ###
        # VLAD Encoding
        #
        # TODO: VLAD encoding port
        #

        return vcd_store_elements

    def generate_frames(self, video_file):
        """
        Process the given video file, producing video level features.

        :param video_file: The video file to process on.
        :type video_file: str

        """
        self._log.info("Extracting frames for: %s", video_file)
        self.mp4_extract_video_frames(video_file)


    def descriptor_generation(self, video_file):
        """
        Process the given video file, producing video level features.

        :param video_file: The video file to process on.
        :type video_file: str

        """
        self._log.info("colordescriptor processing video: %s", video_file)

        # This descriptor has separate output files per video, so creating a
        # sub-work directory to do things in so as not to conflict with other
        # video processing.
        file_prefix, file_key = self.get_video_prefix(video_file)
        video_work_dir = osp.join(self.working_dir, self.DESCRIPTOR_ID,
                                  str(file_prefix), str(file_key))
        self.create_dir(video_work_dir)

        # Defining key file paths
        csift_combined_features_file = osp.join(video_work_dir,
                                                'csift.combined.txt')
        tch_combined_features_file = osp.join(video_work_dir,
                                              'tch.combined.txt')

        # Record space for feature store elements produced by encoding step.
        vcd_store_elements = []

        ###
        # Create Combined files from frame products
        #

        # Different code paths may compress files in different manners. The
        # active path will flip this switch if it compresses the combined files
        # via bzip2
        combined_f_is_bzipped = False

        # If we were provided an old-style working directory and an old-style
        # combined matrix file exists for both csift and tch, skip processing
        # and use these files.
        # TODO: Make this more extension dynamic (i.e. pick up gzipped or normal
        #       txt files too)
        if (self.old_style_work_dir
                and osp.isfile(osp.join(self.old_style_work_dir,
                                        '%s.csift-all.txt.bz2' % file_key))
                and osp.isfile(osp.join(self.old_style_work_dir,
                                        '%s.tcg-all.txt.bz2' % file_key))):
            combined_f_is_bzipped = True
            # Resetting combined file targets to the old-style files.
            # "old-style" as in they're not where we would usually put them.
            csift_combined_features_file = \
                osp.isfile(osp.join(self.old_style_work_dir,
                                    '%s.csift-all.txt.bz2' % file_key))
            tch_combined_features_file = \
                osp.isfile(osp.join(self.old_style_work_dir,
                                    '%s.tcg-all.txt.bz2' % file_key))

        # Skip processing if computed products already exist
        elif not (osp.isfile(csift_combined_features_file)
                  and osp.isfile(tch_combined_features_file)):
            # Extracting video frames needed. Exit with no feature elements if
            # this fails.
            try:
                frm2file, frm2time = \
                    self.mp4_extract_video_frames(video_file)
            except RuntimeError, ex:
                self._log.error("Failed frame extraction! Either process "
                                "failed or no frames were extracted! "
                                "(error: %s) Check up on video: %s",
                                str(ex), video_file)
                return None

            # TODO: Make encapsulation object for video file and accompanying
            #       data structures, so we don't have separate video file and
            #       frame-to-things maps, since they're all tied to each other.
            self._colordescriptor_generation(video_file, frm2file,
                                             video_work_dir,
                                             csift_combined_features_file,
                                             tch_combined_features_file)
        else:
            self._log.info("CSIFT/TCH combined features already computed.")



    def process_video(self, video_file):
        """
        Process the given video file, producing video level features.

        :param video_file: The video file to process on.
        :type video_file: str

        """
        self._log.info("colordescriptor processing video: %s", video_file)

        # This descriptor has separate output files per video, so creating a
        # sub-work directory to do things in so as not to conflict with other
        # video processing.
        file_prefix, file_key = self.get_video_prefix(video_file)
        video_work_dir = osp.join(self.working_dir, self.DESCRIPTOR_ID,
                                  str(file_prefix), str(file_key))
        self.create_dir(video_work_dir)

        # Defining key file paths
        csift_combined_features_file = osp.join(video_work_dir,
                                                'csift.combined.txt')
        tch_combined_features_file = osp.join(video_work_dir,
                                              'tch.combined.txt')

        # Record space for feature store elements produced by encoding step.
        vcd_store_elements = []

        ###
        # Create Combined files from frame products
        #

        # Different code paths may compress files in different manners. The
        # active path will flip this switch if it compresses the combined files
        # via bzip2
        combined_f_is_bzipped = False

        # If we were provided an old-style working directory and an old-style
        # combined matrix file exists for both csift and tch, skip processing
        # and use these files.
        # TODO: Make this more extension dynamic (i.e. pick up gzipped or normal
        #       txt files too)
        if (self.old_style_work_dir
                and osp.isfile(osp.join(self.old_style_work_dir,
                                        '%s.csift-all.txt.bz2' % file_key))
                and osp.isfile(osp.join(self.old_style_work_dir,
                                        '%s.tcg-all.txt.bz2' % file_key))):
            combined_f_is_bzipped = True
            # Resetting combined file targets to the old-style files.
            # "old-style" as in they're not where we would usually put them.
            csift_combined_features_file = \
                osp.isfile(osp.join(self.old_style_work_dir,
                                    '%s.csift-all.txt.bz2' % file_key))
            tch_combined_features_file = \
                osp.isfile(osp.join(self.old_style_work_dir,
                                    '%s.tcg-all.txt.bz2' % file_key))

        # Skip processing if computed products already exist
        elif not (osp.isfile(csift_combined_features_file)
                  and osp.isfile(tch_combined_features_file)):
            # Extracting video frames needed. Exit with no feature elements if
            # this fails.
            try:
                frm2file, frm2time = \
                    self.mp4_extract_video_frames(video_file)
            except RuntimeError, ex:
                self._log.error("Failed frame extraction! Either process "
                                "failed or no frames were extracted! "
                                "(error: %s) Check up on video: %s",
                                str(ex), video_file)
                return None

            # TODO: Make encapsulation object for video file and accompanying
            #       data structures, so we don't have separate video file and
            #       frame-to-things maps, since they're all tied to each other.
            self._colordescriptor_generation(video_file, frm2file,
                                             video_work_dir,
                                             csift_combined_features_file,
                                             tch_combined_features_file)
        else:
            self._log.info("CSIFT/TCH combined features already computed.")

        ###
        # FLANN Encoding
        #
        csift_quantized_file = osp.join(video_work_dir, 'csift.quantized.txt')
        csift_sp_hist_file = osp.join(video_work_dir, 'csift.sp_hist.txt')
        tch_quantized_file = osp.join(video_work_dir, 'tch.quantized.txt')
        tch_sp_hist_file = osp.join(video_work_dir, 'tch.sp_hist.txt')

        csift_feature = self._encode_FLANN(
            csift_combined_features_file, csift_quantized_file,
            csift_sp_hist_file, self.csift_flann_codebook,
            self.csift_flann_file, filein_is_bzipped=combined_f_is_bzipped
        )
        vcd_store_elements.append(
            VCDStoreElement('csift_flann', int(file_key), csift_feature)
        )

        tch_feature = self._encode_FLANN(
            tch_combined_features_file, tch_quantized_file, tch_sp_hist_file,
            self.tch_flann_codebook, self.tch_flann_file,
            filein_is_bzipped=combined_f_is_bzipped
        )
        vcd_store_elements.append(
            VCDStoreElement('tch_flann', int(file_key), tch_feature)
        )

        ###
        # VLAD Encoding
        #
        # TODO: VLAD encoding port
        #

        return vcd_store_elements


    def _colordescriptor_generation(self, video_file, frame2file_map,
                                    working_directory,
                                    output_file_csift, output_file_tch):
        """
        Generate colordescriptor output on the clip-level, generating
        combined files to the given paths. These files wll have the format as
        described by the ``combine_frame_results`` function (see below).

        This method will generate the intermediary per-frame output files from
        the colordescriptor executable as well as the two combined files for the
        two colordescriptor modes to the paths specified.

        :param video_file: The video to generated descriptor matrices from.
        :type video_file: str
        :param frame2file_map: dictionary mapping frame index to the image file
            for that frame as was extracted from the given video file.
        :type frame2file_map: dict of (int, str)
        :param working_directory: The directory in which we will perform and
            store our work.
        :type working_directory: str
        :param output_file_csift: The path to the file to output the combined
            csift descriptor matrix.
        :type output_file_csift: str
        :param output_file_tch: The path to the file to output the combined tch
            descriptor matrix.
        :type output_file_tch: str

        """
        # If either of the combined files exist, don't process that side's
        # material.
        # - This causes us to not even bother collecting potentially existing
        #   computed frame files.
        # - This is for if one side completed but the other didn't due to some
        #   interruption or something.
        compute_csift = not osp.isfile(output_file_csift)
        compute_tch = not osp.isfile(output_file_tch)

        if not (compute_csift or compute_tch):
            # Quick exit when both are already computed.
            self._log.info("Both CSIFT and TCH combined files have already "
                           "been computed.")
            return

        # NOTE:
        # If we are continuing from this point on, then either both need a
        # combined file computed, or just TCH, as its last to get interrupted.
        # Thus, always check TCH frame computation and collect those files for
        # TCH combination.

        w, h, duration, fps = self.mp4_video_properties(video_file)
        file_prefix, file_key = self.get_video_prefix(video_file)

        # For pixel sample grid, we want to take at a maximum of 50,
        # sample points in longest direction with at least a 6 pixel spacing. We
        # will take fewer sample points to ensure the 6 pixel minimum spacing.
        # (magic numbers are
        # a result of tuning)
        sample_size = max(int(math.floor(max(w, h) / 50.0)), 6)

        total_frames = len(frame2file_map)

        # Output files used by this method. These files will also act like stamp
        # files, detailing progress from previous runs. Files will be removed
        # from the disk when the final product of this method has been
        # completed. When this file is present before processing occurs, a total
        # processing skip occurs.
        cd_output_file_pattern = osp.join(working_directory, "%s-%06d.txt")
        cd_log_file_pattern = osp.join(working_directory, "%s-%06d.log")
        # Files will be added to these maps keyed by their frame/index number
        csift_frame_feature_files = {}
        tch_frame_feature_files = {}

        def construct_command(descriptor, input_frame_file, output_file):
            return (self.cdescriptor_exe, input_frame_file,
                    '--detector', 'densesampling',
                    '--ds_spacing', str(sample_size),
                    '--descriptor', descriptor,
                    '--output', output_file)

        self._log.info("starting per-frame descriptor generation "
                       "(%d frame files)", len(frame2file_map))
        dev_null = open('/dev/null', 'w')
        for i, (frame, png_frame) in enumerate(frame2file_map.items()):
            # For the two processing components, if the predicted generated file
            # already exists, skip processing for this frame. Else, remove old
            # tmp file, generated new tmp file, rename to legitimate name.

            # CSIFT execution
            if compute_csift:
                csift_output_file = cd_output_file_pattern % ('csift', frame)
                if not osp.isfile(csift_output_file):
                    csift_log_file = cd_log_file_pattern % ('csift', frame)
                    csift_tmp_file = self.tempify_filename(csift_output_file)
                    if osp.isfile(csift_tmp_file):
                        # remove the file if it exists from previous interrupted
                        # processing
                        os.remove(csift_tmp_file)
                    self._log.info(
                        "[vID:{vid:s} - frm:{cur:d}/{tot:d} ({perc:8.3%})] "
                        "processing csift features"
                        .format(vid=file_key, cur=i,
                                tot=total_frames - 1,
                                perc=float(i * 2) / (total_frames * 2))
                    )
                    self._log.debug('call: %s',
                              ' '.join(construct_command('csift', png_frame,
                                                         csift_tmp_file)))
                    with open(csift_log_file, 'w') as lfile:
                        ret = sub_call(construct_command('csift', png_frame,
                                                         csift_tmp_file),
                                       stdout=lfile, stderr=lfile)
                    if ret != 0:
                        raise RuntimeError("Failed to process colordescriptor "
                                           "csift on frame %s for video %s."
                                           % (png_frame, video_file))
                    try:
                        os.rename(csift_tmp_file, csift_output_file)
                    except:
                        print "Trying to rename: " + csift_tmp_file

                else:
                    self._log.info(
                        "[vID:{vid:s} - frm:{cur:d}/{tot:d} ({perc:8.3%})] "
                        "csift features already processed"
                        .format(vid=file_key, cur=i,
                                tot=total_frames - 1,
                                perc=float(i * 2) / (total_frames * 2))
                    )

                csift_frame_feature_files[frame] = csift_output_file

            # TCH execution
            tch_output_file = cd_output_file_pattern % ('tch', frame)
            if not osp.isfile(tch_output_file):
                tch_log_file = cd_log_file_pattern  %('tch', frame)
                tch_tmp_file = self.tempify_filename(tch_output_file)
                if osp.isfile(tch_tmp_file):
                    os.remove(tch_tmp_file)
                self._log.info(
                    "[vID:{vid:s} - frm:{cur:d}/{tot:d} ({perc:8.3%})] "
                    "processing tch features"
                    .format(vid=file_key, cur=i,
                            tot=total_frames - 1,
                            perc=float(i * 2 + 1) / (total_frames * 2))
                )
                self._log.debug(
                    'call: %s',
                    ' '.join(construct_command('transformedcolorhistogram',
                                               png_frame, tch_tmp_file))
                )
                with open(tch_log_file, 'w') as lfile:
                    ret = sub_call(construct_command('transformedcolorhistogram',
                                                     png_frame, tch_tmp_file),
                                   stdout=lfile, stderr=lfile)
                if ret != 0:
                    raise RuntimeError("Failed to process colordescriptor tch "
                                       "on frame %s for video %s."
                                       % (png_frame, video_file))
                try:
                    os.rename(tch_tmp_file, tch_output_file)
                except:
                    pass

            else:
                self._log.info(
                    "[vID:{vid:s} - frm:{cur:d}/{tot:d} ({perc:8.3%})] "
                    "tch features already processed"
                    .format(vid=file_key, cur=i,
                            tot=total_frames - 1,
                            perc=float(i * 2 + 1) / (total_frames * 2))
                )

            tch_frame_feature_files[frame] = tch_output_file

        # combineResults (local method)
        if compute_csift:
            self._log.debug('combining csift feature matrices -> %s',
                            output_file_csift)
            tmp_file = self.tempify_filename(output_file_csift)
            self._combine_frame_results(csift_frame_feature_files, tmp_file)
            os.rename(tmp_file, output_file_csift)

        self._log.debug('combining tch feature matrices -> %s',
                        output_file_tch)
        tmp_file = self.tempify_filename(output_file_tch)
        self._combine_frame_results(tch_frame_feature_files, tmp_file)
        os.rename(tmp_file, output_file_tch)

    def _combine_frame_results(self, frame_feature_files, output_file):
        """
        Combine descriptor output matrices into a single matrix.

        The combined data matrix representing the all given features has a
        specific format intended for use in quantization (encode_FLANN
        functions):
            [
             [ <frame_num>, <info1>, <info2>, ... <feature vector> ],
             ...
            ]

        :param frame_feature_files: iterable of output files generated by
            color descriptor executable.
        :type frame_feature_files: dict of (int, str)
        :param output_file: The file to output the combined matrix to. If a file
            exists by this name already, it will be overwritten.
        :type output_file: str

        """
        # TODO: add gzip/bzip options / extension recognition?

        with open(output_file, 'w') as output_file:

            for i, ff in sorted(frame_feature_files.items()):
                info, descriptors = DescriptorIO.readDescriptors(ff)

                n = info.shape[0]  # num rows
                data_frame = np.hstack((np.ones((n, 1)) * i,
                                        info[:, 0:2],
                                        descriptors))

                # Continuously adding to the same file with savetxt effectively
                # performs a v-stack operation. '%g' uses the shorter of %e or
                # %f, i.e. exponential or floating point format respectively.
                # TODO: acually vstack arras save to binary file (saves space)
                np.savetxt(output_file, data_frame, fmt='%g')

    def _encode_FLANN(self, combined_file, quantized_file, sphist_file,
                      codebook, flann_file,
                      filein_is_gzipped=False,
                      filein_is_bzipped=False):
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
        :param quantized_file: Path to the file where quantized results are or
            are to be saved.
        :type quantized_file: str
        :param sphist_file: Path to the file where spacial histogram results are
            or are to be saved.
        :type sphist_file: str
        :param codebook: The FLANN codebook to be used for this computation.
        :type codebook: str
        :param flann_file: The FLANN index file to be used for this computation.
        :type flann_file: str
        :param filein_is_gzipped: Whether the input combined file is gzipped or
            not.
        :type filein_is_gzipped: bool
        :param filein_is_bzipped: Whether the input combined file is bzipped or
            not.
        :type filein_is_bzipped: boot
        :return: A 1D numpy array (vector) representing the video-level feature.
        :rtype: numpy.ndarray

        """
        if not osp.isfile(quantized_file):
            self._log.debug('building FLANN quantized file')
            tmp_file = self.tempify_filename(quantized_file)
            encode_FLANN.quantizeResults2(combined_file, tmp_file,
                                          codebook, flann_file,
                                          filein_is_gzipped,
                                          filein_is_bzipped)
            try:
                os.rename(tmp_file, quantized_file)
            except:
                pass
        else:
            self._log.debug('existing quantized file found')

        if not osp.isfile(sphist_file):
            self._log.debug('building FLANN spacial pyramid-')
            tmp_file = self.tempify_filename(sphist_file)
            encode_FLANN.build_sp_hist_(quantized_file, tmp_file)
            os.rename(tmp_file, sphist_file)
        else:
            self._log.debug('existing sphist file found')

        # Histogram file will consist of 8 vectors. Unified vector is all of
        # those h-stacked.

        # Result of build_sp_hist is an 8xN matrix, where each row is a
        # clip-level feature for a spacial region. Final feature product
        # will be a 4 subset of these 8 vectors h-stacked. Voila, clip level feature!
        _hist_sp = np.hstack(np.loadtxt(sphist_file)[[0,5,6,7], :])
        hist_sp = _hist_sp / np.sum(_hist_sp)
        return hist_sp
