"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from Cheetah.Template import Template as CTemplate
import numpy as np
import os
import os.path as osp
import shutil
import subprocess
import time

from ...VCDStore import VCDStoreElement

from .. import VCDWorkerInterface

from _sunattributes import load_attribute_list
from _sunattributes.calibration_fittings import load_calibration_fittings


class sun_attributes (VCDWorkerInterface):

    DESCRIPTOR_ID = 'sunattributes'

    @classmethod
    def generate_config(cls, config=None):
        config = super(sun_attributes, cls).generate_config(config)

        sect = cls.DESCRIPTOR_ID
        if not config.has_section(sect):
            config.add_section(
                sect,
                "Options for the SUN Attributes VCD\n"
                "\n"
                "Creates store elements with the following descriptor IDs:\n"
                "   - sunattributes_raw_avg\n"
                "   - sunattributes_raw_max\n"
                "   - sunattributes_calib_avg\n"
                "   - sunattributes_calib_max\n"
            )

        config.set(sect, 'sa_source_directory', '',
                   "Directory where required sun_attributes architecture is "
                   "located. i.e. the directory that contains such other "
                   "directories as 'KWVision' and, 'SUN_source_code_v2',"
                   "'vlfeat-0.9.16', etc.")

        config.set(sect, 'mcr_root', '',
                   "Directory location of the MATLAB MCR root directory is.")

        config.set(sect, 'sun_attributes_runner',
                   '%(sa_source_directory)s/deploy/compute_sun_attributes_multi'
                   '/src/run_compute_sun_attributes_multi.sh',
                   "The executable wrapper script for the sun attributes "
                   "executable. With a properly configured "
                   "'sa_source_directory' option, this should not have to be "
                   "changed.")

        config.set(sect, 'config_xml_template',
                   osp.join(osp.dirname(__file__), '_sunattributes',
                            'sa_config.xml.tmpl'),
                   "Sun attributes configuration file template. This path "
                   "should probably not be modified.")

        config.set(sect, 'matlab_run_timeout_minutes', '600.0',
                   'Maximum time MATLAB processes is allowed to run. After '
                   'this period we will kill the subprocess and assume video '
                   'run failure.')

        config.set(sect, 'timeout_poll_interval', '0.01',
                   'Poll interval to check for process run timeout.')

    def __init__(self, config, working_dir, image_root):
        """
        :param config: Configuration object for the descriptor worker.
        :type config: SafeConfigCommentParser
        :param working_dir: The directory where work will be stored.
        :type working_dir: str
        :param image_root: Working image output and storage root directory. If
            None is provided, a path will be determined automatically within the
            given working directory.
        :type image_root: str

        """
        super(sun_attributes, self).__init__(config, working_dir, image_root)

        sect = self.DESCRIPTOR_ID
        self._sa_src_dir = config.get(sect, "sa_source_directory")
        self._mcr_root = config.get(sect, "mcr_root")
        self._sa_runner = config.get(sect, "sun_attributes_runner")
        self._config_tmpl = config.get(sect, "config_xml_template")
        self._matlab_run_timeout_seconds = \
            config.getfloat(sect, 'matlab_run_timeout_minutes') * 60.0
        self._timeout_poll_interval = \
            config.getfloat(sect, 'timeout_poll_interval')

        # Loading calibration matrices
        self._calib_map = load_calibration_fittings()

        # The attributes, in order, as output by SUN Attributes executable for a
        # frame.
        self._attribute_list = load_attribute_list()

        self._log.debug('sa_source_directory: %s', self._sa_src_dir)
        self._log.debug('mcr_root: %s', self._mcr_root)
        self._log.debug('sun_attributes_runner: %s', self._sa_runner)
        self._log.debug('config_xml_template:5 %s', self._config_tmpl)
        self._log.debug('matlab_run_timeout_seconds: %f',
                        self._matlab_run_timeout_seconds)

    def process_video(self, video_file):
        """
        video processing step for sun_attributes descriptor.

        :param video_file: Video file to process
        :type video_file: str
        :return: Tuple of VCDStoreElements generated
        :rtype: tuple of VCDStoreElement

        """
        # Steps overview:
        #   - extract required image frames from video
        #   - create video working directory if not there (vwd)
        #   - generated an xml configuration file for the video in vwd
        #   - generated work list (frame input/output locations)
        #   - combined generated feature vectors (element-wise average)
        #   - generated VCDStoreElement

        # Key files for processing
        pfx, key = self.get_video_prefix(video_file)
        video_work_dir = osp.join(self.working_dir, self.DESCRIPTOR_ID,
                                  pfx, key)
        self.create_dir(video_work_dir)

        combined_file = \
            osp.join(video_work_dir, 'raw.features.npy')
        combined_file_calib = \
            osp.join(video_work_dir, 'calib.features.npy')
        raw_avg_feature_file = \
            osp.join(video_work_dir, 'raw.vfeature.avg.npy')
        raw_max_feature_file = \
            osp.join(video_work_dir, 'raw.vfeature.max.npy')
        calib_avg_feature_file = \
            osp.join(video_work_dir, 'calib.vfeature.avg.npy')
        calib_max_feature_file = \
            osp.join(video_work_dir, 'calib.vfeature.max.npy')

        def handle_checkpoint(checkpoint_file, func, args):
            """
            Handle score computation function and check-pointing the step to
            file to prevent repeat computation across runs.
            """
            if not osp.isfile(checkpoint_file):
                scores = func(*args)
                tmp = self.tempify_filename(checkpoint_file)
                with open(tmp, 'w') as ofile:
                    # noinspection PyTypeChecker
                    np.save(ofile, scores)
                os.rename(tmp, checkpoint_file)
            else:
                scores = np.load(checkpoint_file)
            return scores

        try:
            raw_scores = handle_checkpoint(combined_file,
                                           self._generate_attributes,
                                           (video_file, video_work_dir))
        except RuntimeError, ex:
            self._log.error("Couldn't computed raw scores due to RuntimeError, "
                            "skipping processing for video: %s\n"
                            "(error: %s)", video_file, str(ex))
            return None

        calib_scores = handle_checkpoint(combined_file_calib,
                                         self._calibrate_attributes,
                                         (raw_scores,))

        avg = lambda v: sum(v) / float(len(v))

        raw_max_scores = handle_checkpoint(raw_max_feature_file,
                                           self._aggregate_columnwise,
                                           (raw_scores, max))
        raw_avg_scores = handle_checkpoint(raw_avg_feature_file,
                                           self._aggregate_columnwise,
                                           (raw_scores, avg))
        calib_max_scores = handle_checkpoint(calib_max_feature_file,
                                             self._aggregate_columnwise,
                                             (calib_scores, max))
        calib_avg_scores = handle_checkpoint(calib_avg_feature_file,
                                             self._aggregate_columnwise,
                                             (calib_scores, avg))

        # construct 4 elements for storage
        ikey = int(key)
        rm_se = VCDStoreElement('sunattributes_raw_max',
                                ikey, raw_max_scores)
        ra_se = VCDStoreElement('sunattributes_raw_avg',
                                ikey, raw_avg_scores)
        cm_se = VCDStoreElement('sunattributes_calibrated_max',
                                ikey, calib_max_scores)
        ca_se = VCDStoreElement('sunattributes_calibrated_avg',
                                ikey, calib_avg_scores)

        return rm_se, ra_se, cm_se, ca_se

    def _generate_attributes(self, video_file, video_work_dir):
        """
        Run actual sun_attributes code on video, outputting a file that is the
        concatenation of frame feature products (checkpoint file) and returning
        the resultant matrix. The output file will be a numpy binary file
        storing a 2D array object.

        If the output file already exists, we will assume that the work as
        already been completed and will load and return that file's matrix.

        :raises RuntimeError: Couldn't create raw scores due to error in
            runtime. This can either be due to frame extraction failure or
            failure at running the SUN attributes executable.

        :param video_file: Video for processing
        :type video_file: str
        :param video_work_dir: Directory to store work in
        :type video_work_dir: str
        :return: 2D numpy array object that is the combined frame feature
            vectors, or attribute scores. If None is returned, that means scores
            couldn't be produced on this video.
        :rtype: numpy.ndarray

        """
        # video frame extraction
        #
        # extracting jpb because the MATLAB code only works with jpg files
        # apparently...
        #
        # exiting failure if no frames extracted.
        #
        frame2img, _ = self.mp4_extract_video_frames(video_file, 'jpg')

        # run output vars
        f_result_output_dir = osp.join(video_work_dir, 'frame_results')
        self.create_dir(f_result_output_dir)
        output_files = {}
        tmp_output_files = {}  # tmp version of output_files map
        of_not_present = []  # frames for which no non-temp output present

        # Predict output files we will need for images specified/produced
        # If one or more of the predicted output files doesn't exist in
        # their predicted location, queue it to be processed.
        for frame, img_file in sorted(frame2img.items()):
            output_files[frame] = osp.join(f_result_output_dir,
                                           'frm%06d.features' % frame)
            tmp_output_files[frame] = \
                self.tempify_filename(output_files[frame])
            if not osp.isfile(output_files[frame]):
                of_not_present.append(frame)

        # Only create a work file and process if anything actually needs to
        # be processed
        if of_not_present:
            # remove old temp files if present (i don't know what the MATLAB
            # code might do with a partially written file, i.e. if it will
            # append to it or not)
            self._log.debug("removing pre-existing temp frame work files")
            for f in tmp_output_files.values():
                if osp.isfile(f):
                    os.remove(f)

            # Create actual work file, overwriting any that might already be
            # present.
            work_file = osp.join(video_work_dir, 'sa_work_file.txt')
            with open(work_file, 'w') as wfile:
                for frame in of_not_present:
                    wfile.write("%s,%s\n" % (frame2img[frame],
                                             tmp_output_files[frame]))

            # Generating configuration file from template, overwriting an
            # existing one if present (things may be different).
            config_file = osp.join(video_work_dir, 'sa_config.xml')
            sa_tmp_folder = \
                self.create_dir(osp.join(video_work_dir, 'sa_tmp'))
            sl = {'sa_src_dir': self._sa_src_dir,
                  'working_dir': sa_tmp_folder}
            with open(config_file, 'w') as cfile:
                cfile.write(str(CTemplate(file=self._config_tmpl,
                                          searchList=sl)))

            run_args = [self._sa_runner, self._mcr_root, work_file,
                        config_file]
            self._log.debug("call: %s", ' '.join(run_args))
            log_file = osp.join(video_work_dir, 'last_run.log')
            with open(log_file, 'w') as lf:
                # MATLAB is known to rarely timeout indefinitely. Using ex
                start = time.time()
                p = subprocess.Popen(run_args, stdout=lf, stderr=lf)
                while p.poll() is None:
                    # check if timeout exceeded, kicking the bucket if so
                    if (time.time() - start) > self._matlab_run_timeout_seconds:
                        p.terminate()
                        raise RuntimeError(
                            "MATLAB processing exceeded configured timeout "
                            "(%.1f minutes). Check run log file for possible "
                            "details as to why: %s",
                            self._matlab_run_timeout_seconds / 60.0, log_file
                        )
                    time.sleep(self._timeout_poll_interval)

            if p.returncode != 0:
                raise RuntimeError("Failed to process sun_attributes "
                                   "for video %s with return code %s. "
                                   "Check log file: %s"
                                   % (video_file, str(p.returncode), log_file))

            # Run successful, check that configured tmp files exist and
            # rename them to the official output names.
            for frame in of_not_present:
                if not osp.isfile(tmp_output_files[frame]):
                    raise RuntimeError("sun_attributes failed to produce "
                                       "output for requested frame %d. "
                                       "Check that it is actually present "
                                       "in the generated work file (%s) "
                                       "as well as check the log file (%s)."
                                       % (frame, work_file, log_file))
                os.rename(tmp_output_files[frame], output_files[frame])

            # Remove temp work files generated by sun_attributes
            self._log.debug("removing sa temp working directory")
            shutil.rmtree(sa_tmp_folder)

        ###
        # Combination
        #
        # First, need to construct the combined matrix, which is just each
        # frame's results stacked from top to bottom in frame order.
        #
        self._log.debug("creating raw combined features matrix")
        combined_matrix = []
        for frm, features_file in sorted(output_files.items()):
            self._log.debug("-> adding frame %d : %s", frm, features_file)
            combined_matrix.append(np.loadtxt(features_file))
        combined_matrix = np.array(combined_matrix)  # make 2D array of list

        return combined_matrix

    def _calibrate_attributes(self, raw_scores):
        """
        Calibrate raw scores matrix with stored calibration data.

        If the checkpoint file already exists, we assume calibration has already
        occurred, and will load and return the calibration matrix from file.

        :param raw_scores: Raw score matrix to calibrate.
        :type raw_scores: numpy.ndarray
        :return: Calibrated scores matrix. This will be the same shape as the
            input raw scores matrix.
        :rtype: numpy.ndarray

        """
        assert raw_scores.shape[1] == len(self._attribute_list), \
            "Raw score matrix dimensions did not match expected number of " \
            "attributes!"
        assert len(self._calib_map.keys()) == len(self._attribute_list), \
            "Calibration map dimension did not match expected number of " \
            "attributes!"

        mapped_scores = np.zeros(raw_scores.shape)

        # For each calibration mapping, transform the corresponding column
        # values in the raw_scores matrix.
        self._log.info("Calibrating raw scores")
        for dim, attr in enumerate(self._attribute_list):
            score_mapping = self._calib_map[attr]

            for row in range(len(raw_scores)):
                s = raw_scores[row][dim]
                if s >= score_mapping[0][0]:
                    # greater than mapping interval range
                    mapped_scores[row][dim] = score_mapping[0][1]
                elif s <= score_mapping[-1][0]:
                    # less than mapping interval range
                    mapped_scores[row][dim] = score_mapping[-1][1]
                else:
                    for i in range(1, len(score_mapping)-1):
                        if (s <= score_mapping[i][0]) & (s >= score_mapping[i+1][0]):
                            ratio = (s - score_mapping[i][0]) / (score_mapping[i][0] - score_mapping[i+1][0])
                            mapped_scores[row][dim] = ratio * (score_mapping[i][1] - score_mapping[i+1][1]) + score_mapping[i+1][1]

        return mapped_scores

    def _aggregate_columnwise(self, matrix, func):
        """
        for the given matrix, compute the element-wise function over each
        column in the matrix, returning the resultant row vector. ``func``
        should be a function that takes an iterable and returns a single
        floating point value.

        :param matrix: 2D numpy array to compute over
        :type matrix: numpy.ndarray
        :param func: Aggregation function
        :type func: types.FunctionType
        :return:  Numpy vector after column-wise aggregation
        :rtype: numpy.ndarray

        """
        return np.array(tuple(func(matrix[:, i])
                              for i in xrange(matrix.shape[1])))
