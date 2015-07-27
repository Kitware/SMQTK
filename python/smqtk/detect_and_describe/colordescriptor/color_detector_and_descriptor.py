import abc
import logging
import multiprocessing
import multiprocessing.pool
import numpy
import os.path as osp

from smqtk.detect_and_describe import DetectAndDescribe
from smqtk.utils import safe_create_dir, SimpleTimer
from smqtk.utils.string_utils import partition_string

from . import utils

class ColorDetectAndDescribe_Base (DetectAndDescribe):
    PROC_COLORDESCRIPTOR = "colorDescriptor"

    # TODO -- where is this supposed to get set?
    PARALLEL = 2

    def __init__(self, work_directory):
        """
        Initialize a new ColorDetectAndDescribe interface instance.
        :param work_directory: Path to the directory in which to place temporary/working
            files. Relative paths are treated relative to 'smqtk_config.WORK_DIR'.
        :type work_directory: str | unicode
        """
        ### TODO -- spatial
        self._work_directory = work_directory

        self._log = logging.getLogger('.'.join([ColorDetectAndDescribe_Base.__module__,
            ColorDetectAndDescribe_Base.__name__]))

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def temp_dir(self):
        return safe_create_dir(osp.join(self._work_directory, 'temp_files'))
    
    @abc.abstractmethod
    def descriptor_type(self):
        """
        :return: String descriptor type as used by colorDescriptor
        :rtype: str
        """
        return

    def detect_and_describe(self, data):
        pass

    @classmethod
    def is_usable(cls):
        """
        Check whether this descriptor is available for use.

        :return: Boolean determination of whether this implementation is usable.
        :rtype: bool

        """
        log = logging.getLogger('.'.join([ColorDetectAndDescribe_Base.__module__,
                                          ColorDetectAndDescribe_Base.__name__,
                                          "is_usable"]))

        if not hasattr(ColorDetectAndDescribe_Base, "_is_usable_cache"):
            # Check for colorDescriptor executable on the path
            import subprocess
            try:
                # This should try to print out the CLI options return with code
                # 1.
                subprocess.call(['colorDescriptor', '-h'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
            except OSError:
                log.warn("Could not locate colorDescriptor executable. Make "
                         "sure that its on the PATH! See "
                         "smqtk/content_description/colordescriptor/INSTALL.md "
                         "for help.")
                ColorDetectAndDescribe_Base._is_usable_cache = False

            # Checking if DescriptorIO is importable
            if not utils.has_colordescriptor_module():
                log.warn("Could not import DescriptorIO. Make sure that the "
                         "colorDescriptor package is on the PYTHONPATH! See "
                         "smqtk/content_description/colordescriptor/INSTALL.md "
                         "for help.")
                ColorDetectAndDescribe_Base._is_usable_cache = False

            ColorDetectAndDescribe_Base._is_usable_cache = True

        return ColorDetectAndDescribe_Base._is_usable_cache

    def _get_checkpoint_dir(self, data):
        """
        The directory that contains checkpoint material for a given data element

        :param data: Data element
        :type data: smqtk.data_rep.DataElement

        :return: directory path
        :rtype: str

        """
        d = osp.join(self._work_directory, *partition_string(data.md5(), 8))
        safe_create_dir(d)
        return d

    def _get_standard_info_descriptors_filepath(self, data, frame=None):
        """
        Get the standard path to a data element's computed descriptor output,
        which for colorDescriptor consists of two matrices: feat and descriptors

        :param data: Data element
        :type data: smqtk.data_rep.DataElement

        :param frame: frame within the data file
        :type frame: int

        :return: Paths to feat and descriptor checkpoint numpy files
        :rtype: (str, str)

        """
        d = self._get_checkpoint_dir(data)
        if frame is not None:
            return (
                osp.join(d, "%s.feat.%d.npy" % (data.md5(), frame)),
                osp.join(d, "%s.descriptors.%d.npy" % (data.md5(), frame))
            )
        else:
            return (
                osp.join(d, "%s.feat.npy" % data.md5()),
                osp.join(d, "%s.descriptors.npy" % data.md5())
            )

    def _get_checkpoint_feature_file(self, data):
        """
        Return the standard path to a data element's computed feature checkpoint
        file relative to our current working directory.

        :param data: Data element
        :type data: smqtk.data_rep.DataElement

        :return: Standard path to where the feature checkpoint file for this
            given data element.
        :rtype: str

        """
        if self._use_sp:
            return osp.join(self._get_checkpoint_dir(data),
                            "%s.feature.sp.npy" % data.md5())
        else:
            return osp.join(self._get_checkpoint_dir(data),
                            "%s.feature.npy" % data.md5())


class ColorDetectAndDescribe_Image (ColorDetectAndDescribe_Base):
    def detect_and_describe(self, data_set, **kwargs):
        """
        Generate feature and descriptor matrices based on ingest type.

        :param data_set: Iterable of data elements to generate combined info
            and descriptor matrices for.
        :type item_iter: collections.Set[smqtk.data_rep.DataElement]

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        if not data_set:
            raise ValueError("No data given to process.")

        # Safety checks
        data_set = super(ColorDetectAndDescribe_Base, self).detect_and_describe(data_set)

        inf = float('inf')
        descriptor_limit = kwargs.get('limit', inf)
        per_item_limit = numpy.floor(float(descriptor_limit) / len(data_set))

        if len(data_set) == 1:
            self._log.debug("Building descriptor matrices for a data_set of length 1")
            # because an iterable doesn't necessarily have a next() method
            di = iter(data_set).next()
            # Check for checkpoint files
            feat_fp, desc_fp = \
                self._get_standard_info_descriptors_filepath(di)
            # Save out data bytes to temporary file
            temp_img_filepath = di.write_temp(self.temp_dir)
            try:
                # Generate descriptors
                utils.generate_descriptors(
                    self.PROC_COLORDESCRIPTOR, temp_img_filepath,
                    self.descriptor_type(), feat_fp, desc_fp, per_item_limit
                )
            finally:
                # clean temp file
                di.clean_temp()
            return numpy.load(feat_fp), numpy.load(desc_fp)
            
        else:
            self._log.debug("Building descriptor matrices for a data_set of length %s" % len(data_set))
            # compute and V-stack matrices for all given images
            pool = multiprocessing.Pool(processes=self.PARALLEL)

            # Mapping of UID to tuple containing:
            #   (feat_fp, desc_fp, async processing result, tmp_clean_method)
            r_map = {}
            with SimpleTimer("Computing descriptors async...", self._log.debug):
                for di in data_set:
                    # Creating temporary image file from data bytes
                    tmp_img_fp = di.write_temp(self.temp_dir)

                    feat_fp, desc_fp = \
                        self._get_standard_info_descriptors_filepath(di)
                    args = (self.PROC_COLORDESCRIPTOR, tmp_img_fp,
                            self.descriptor_type(), feat_fp, desc_fp)
                    r = pool.apply_async(utils.generate_descriptors, args)
                    r_map[di.uuid()] = (feat_fp, desc_fp, r, di.clean_temp)
            pool.close()

            # Pass through results from descriptor generation, aggregating
            # matrix shapes.
            # - Transforms r_map into:
            #       UID -> (feat_fp, desc_fp, starting_row, SubSampleIndices)
            self._log.debug("Constructing information for super matrices...")
            s_keys = sorted(r_map.keys())
            running_height = 0  # info and desc heights congruent

            f_width = None
            d_width = None

            for uid in s_keys:
                ffp, dfp, r, tmp_clean_method = r_map[uid]

                # descriptor generation may have failed for this ingest UID
                try:
                    f_shape, d_shape = r.get()
                except RuntimeError, ex:
                    self._log.warning("Descriptor generation failed for "
                                     "UID[%s], skipping its inclusion in "
                                     "model: %s", uid, str(ex))
                    r_map[uid] = None
                    continue
                finally:
                    # Done with image file, so remove from filesystem
                    tmp_clean_method()

                if d_width is None and d_shape[0] != 0:
                    f_width = f_shape[1]
                    d_width = d_shape[1]

                # skip this result if it generated no descriptors
                if d_shape[1] == 0:
                    continue

                ssi = None
                if f_shape[0] > per_item_limit:
                    # pick random indices to subsample down to size limit
                    ssi = sorted(
                        numpy.random.permutation(f_shape[0])[:per_item_limit]
                    )

                # Only keep this if any descriptors were generated
                r_map[uid] = (ffp, dfp, running_height, ssi)
                running_height += min(f_shape[0], per_item_limit)
            pool.join()

            # Asynchronously load files, inserting data into master matrices
            self._log.debug("Building super matrices...")
            master_feat = numpy.zeros((running_height, f_width), dtype=float)
            master_desc = numpy.zeros((running_height, d_width), dtype=float)
            tp = multiprocessing.pool.ThreadPool(processes=self.PARALLEL)
            for uid in s_keys:
                if r_map[uid]:
                    ffp, dfp, sR, ssi = r_map[uid]
                    tp.apply_async(ColorDetectAndDescribe_Image._thread_load_matrix,
                                   args=(ffp, master_feat, sR, ssi))
                    tp.apply_async(ColorDetectAndDescribe_Image._thread_load_matrix,
                                   args=(dfp, master_desc, sR, ssi))
            tp.close()
            tp.join()
            return master_feat, master_desc

    def valid_content_types(self):
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        :rtype: set[str]
        """
        return {'image/bmp', 'image/jpeg', 'image/png', 'image/tiff'}

    @staticmethod
    def _thread_load_matrix(filepath, m, sR, subsample=None):
        """
        load a numpy matrix from ``filepath``, inserting the loaded matrix into
        ``m`` starting at the row ``sR``.

        If subsample has a value, it will be a list if indices to
        """
        n = numpy.load(filepath)
        if subsample:
            n = n[subsample, :]
        m[sR:sR+n.shape[0], :n.shape[1]] = n    


class ColorDetectAndDescribe_Video (ColorDetectAndDescribe_Base):
    def detect_and_describe(self, data_set, **kwargs):
        """
        Generate feature and descriptor matrices based on ingest type.

        :param data_set: Iterable of data elements to generate combined info
            and descriptor matrices for.
        :type item_iter: collections.Set[smqtk.data_rep.DataElement]

        :param limit: Limit the number of descriptor entries to this amount.
        :type limit: int

        :return: Combined info and descriptor matrices for all base images
        :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

        """
        descriptor_limit = kwargs.get('limit', float('inf'))
        # With videos, an "item" is one video, so, collect for a while video
        # as normal, then subsample from the full video collection.
        per_item_limit = numpy.floor(float(descriptor_limit) / len(data_set))

        # If an odd number of jobs, favor descriptor extraction
        if self.PARALLEL:
            descr_parallel = int(max(1, math.ceil(self.PARALLEL/2.0)))
            extract_parallel = int(max(1, math.floor(self.PARALLEL/2.0)))
        else:
            cpuc = multiprocessing.cpu_count()
            descr_parallel = int(max(1, math.ceil(cpuc/2.0)))
            extract_parallel = int(max(1, math.floor(cpuc/2.0)))

        # For each video, extract frames and submit colorDescriptor processing
        # jobs for each frame, combining all results into a single matrix for
        # return.
        pool = multiprocessing.Pool(processes=descr_parallel)

        # Mapping of [UID] to [frame] to tuple containing:
        #   (feat_fp, desc_fp, async processing result)
        r_map = {}
        with SimpleTimer("Extracting frames and submitting descriptor jobs...",
                         self.log.debug):
            for di in data_set:
                r_map[di.uuid()] = {}
                tmp_vid_fp = di.write_temp(self.temp_dir)
                p = dict(self.FRAME_EXTRACTION_PARAMS)
                vmd = get_metadata_info(tmp_vid_fp)
                p['second_offset'] = vmd.duration * p['second_offset']
                p['max_duration'] = vmd.duration * p['max_duration']
                fm = video_utils.ffmpeg_extract_frame_map(
                    tmp_vid_fp,
                    parallel=extract_parallel,
                    **p
                )

                # Compute descriptors for extracted frames.
                for frame, imgPath in fm.iteritems():
                    feat_fp, desc_fp = \
                        self._get_standard_info_descriptors_filepath(di, frame)
                    r = pool.apply_async(
                        utils.generate_descriptors,
                        args=(self.PROC_COLORDESCRIPTOR, imgPath,
                              self.descriptor_type(), feat_fp, desc_fp)
                    )
                    r_map[di.uuid()][frame] = (feat_fp, desc_fp, r)

                # Clean temporary video file file while computing descriptors
                # This does not remove the extracted frames that the underlying
                #   detector/descriptor is working on.
                di.clean_temp()
        pool.close()

        # Each result is a tuple of two ndarrays: info and descriptor matrices
        with SimpleTimer("Collecting shape information for super matrices...",
                         self.log.debug):
            running_height = 0

            f_width = None
            d_width = None

            # Transform r_map[uid] into:
            #   (info_mat_files, desc_mat_files, sR, ssi_list)
            #   -> files in frame order
            uids = sorted(r_map)
            for uid in uids:
                video_num_desc = 0
                video_feat_mat_fps = []  # ordered list of frame info mat files
                video_desc_mat_fps = []  # ordered list of frame desc mat files
                for frame in sorted(r_map[uid]):
                    ffp, dfp, r = r_map[uid][frame]

                    # Descriptor generation may have failed for this UID
                    try:
                        f_shape, d_shape = r.get()
                    except RuntimeError, ex:
                        self.log.warning('Descriptor generation failed for '
                                         'frame %d in video UID[%s]: %s',
                                         frame, uid, str(ex))
                        r_map[uid] = None
                        continue

                    if d_width is None and d_shape[0] != 0:
                        f_width = f_shape[1]
                        d_width = d_shape[1]

                    # Skip if there were no descriptors generated for this
                    # frame
                    if d_shape[1] == 0:
                        continue

                    video_feat_mat_fps.append(ffp)
                    video_desc_mat_fps.append(dfp)
                    video_num_desc += d_shape[0]

                # If combined descriptor height exceeds the per-item limit,
                # generate a random subsample index list
                ssi = None
                if video_num_desc > per_item_limit:
                    ssi = sorted(
                        numpy.random.permutation(video_num_desc)[:per_item_limit]
                    )
                    video_num_desc = len(ssi)

                r_map[uid] = (video_feat_mat_fps, video_desc_mat_fps,
                              running_height, ssi)
                running_height += video_num_desc
        pool.join()
        del pool

        with SimpleTimer("Building master descriptor matrices...",
                         self.log.debug):
            master_feat = numpy.zeros((running_height, f_width), dtype=float)
            master_desc = numpy.zeros((running_height, d_width), dtype=float)
            tp = multiprocessing.pool.ThreadPool(processes=self.PARALLEL)
            for uid in uids:
                feat_fp_list, desc_fp_list, sR, ssi = r_map[uid]
                tp.apply_async(ColorDescriptor_Video._thread_load_matrices,
                               args=(master_feat, feat_fp_list, sR, ssi))
                tp.apply_async(ColorDescriptor_Video._thread_load_matrices,
                               args=(master_desc, desc_fp_list, sR, ssi))
            tp.close()
            tp.join()

        return master_feat, master_desc

    @staticmethod
    def _thread_load_matrices(m, file_list, sR, subsample=None):
        """
        load numpy matrices from files in ``file_list``, concatenating them
        vertically. If a list of row indices is provided in ``subsample`` we
        subsample those rows out of the concatenated matrix. This matrix is then
        inserted into ``m`` starting at row ``sR``.
        """
        c = numpy.load(file_list[0])
        for i in range(1, len(file_list)):
            c = numpy.vstack((c, numpy.load(file_list[i])))
        if subsample:
            c = c[subsample, :]
        m[sR:sR+c.shape[0], :c.shape[1]] = c


# Begin automatic class type creation
valid_descriptor_types = [
    'rgbhistogram',
    'opponenthistogram',
    'huehistogram',
    'nrghistogram',
    'transformedcolorhistogram',
    'colormoments',
    'colormomentinvariants',
    'sift',
    'huesift',
    'hsvsift',
    'opponentsift',
    'rgsift',
    'csift',
    'rgbsift',
]


def _create_image_descriptor_class(descriptor_type_str):
    """
    Create and return a ColorDescriptor class that operates over Image files
    using the given descriptor type.
    """
    assert descriptor_type_str in valid_descriptor_types, \
        "Given ColorDescriptor type was not valid! Given: %s. Expected one " \
        "of: %s" % (descriptor_type_str, valid_descriptor_types)

    # noinspection PyPep8Naming
    class _cd_image_impl (ColorDetectAndDescribe_Image):
        def descriptor_type(self):
            """
            :rtype: str
            """
            return descriptor_type_str

    _cd_image_impl.__name__ = "ColorDetectAndDescribe_Image_%s" % descriptor_type_str
    return _cd_image_impl


def _create_video_descriptor_class(descriptor_type_str):
    """
    Create and return a ColorDescriptor class that operates over Video files
    using the given descriptor type.
    """
    assert descriptor_type_str in valid_descriptor_types, \
        "Given ColorDescriptor type was not valid! Given: %s. Expected one " \
        "of: %s" % (descriptor_type_str, valid_descriptor_types)

    # noinspection PyPep8Naming
    class _cd_video_impl (ColorDetectAndDescribe_Video):
        def descriptor_type(self):
            """
            :rtype: str
            """
            return descriptor_type_str

    _cd_video_impl.__name__ = "ColorDetectAndDescribe_Video_%s" % descriptor_type_str
    return _cd_video_impl


# In order to allow multiprocessing, class types must be concretely assigned to
# variables in the module. Dynamic generation causes issues with pickling (the
# default data transmission protocol).

ColorDetectAndDescribe_Image_rgbhistogram = _create_image_descriptor_class('rgbhistogram')
ColorDetectAndDescribe_Image_opponenthistogram = _create_image_descriptor_class('opponenthistogram')
ColorDetectAndDescribe_Image_huehistogram = _create_image_descriptor_class('huehistogram')
ColorDetectAndDescribe_Image_nrghistogram = _create_image_descriptor_class('nrghistogram')
ColorDetectAndDescribe_Image_transformedcolorhistogram = _create_image_descriptor_class('transformedcolorhistogram')
ColorDetectAndDescribe_Image_colormoments = _create_image_descriptor_class('colormoments')
ColorDetectAndDescribe_Image_colormomentinvariants = _create_image_descriptor_class('colormomentinvariants')
ColorDetectAndDescribe_Image_sift = _create_image_descriptor_class('sift')
ColorDetectAndDescribe_Image_huesift = _create_image_descriptor_class('huesift')
ColorDetectAndDescribe_Image_hsvsift = _create_image_descriptor_class('hsvsift')
ColorDetectAndDescribe_Image_opponentsift = _create_image_descriptor_class('opponentsift')
ColorDetectAndDescribe_Image_rgsift = _create_image_descriptor_class('rgsift')
ColorDetectAndDescribe_Image_csift = _create_image_descriptor_class('csift')
ColorDetectAndDescribe_Image_rgbsift = _create_image_descriptor_class('rgbsift')

ColorDetectAndDescribe_Video_rgbhistogram = _create_video_descriptor_class('rgbhistogram')
ColorDetectAndDescribe_Video_opponenthistogram = _create_video_descriptor_class('opponenthistogram')
ColorDetectAndDescribe_Video_huehistogram = _create_video_descriptor_class('huehistogram')
ColorDetectAndDescribe_Video_nrghistogram = _create_video_descriptor_class('nrghistogram')
ColorDetectAndDescribe_Video_transformedcolorhistogram = _create_video_descriptor_class('transformedcolorhistogram')
ColorDetectAndDescribe_Video_colormoments = _create_video_descriptor_class('colormoments')
ColorDetectAndDescribe_Video_colormomentinvariants = _create_video_descriptor_class('colormomentinvariants')
ColorDetectAndDescribe_Video_sift = _create_video_descriptor_class('sift')
ColorDetectAndDescribe_Video_huesift = _create_video_descriptor_class('huesift')
ColorDetectAndDescribe_Video_hsvsift = _create_video_descriptor_class('hsvsift')
ColorDetectAndDescribe_Video_opponentsift = _create_video_descriptor_class('opponentsift')
ColorDetectAndDescribe_Video_rgsift = _create_video_descriptor_class('rgsift')
ColorDetectAndDescribe_Video_csift = _create_video_descriptor_class('csift')
ColorDetectAndDescribe_Video_rgbsift = _create_video_descriptor_class('rgbsift')


cd_type_list = [
    ColorDetectAndDescribe_Image_rgbhistogram,
    ColorDetectAndDescribe_Video_rgbhistogram,
    ColorDetectAndDescribe_Image_opponenthistogram,
    ColorDetectAndDescribe_Video_opponenthistogram,
    ColorDetectAndDescribe_Image_huehistogram,
    ColorDetectAndDescribe_Video_huehistogram,
    ColorDetectAndDescribe_Image_nrghistogram,
    ColorDetectAndDescribe_Video_nrghistogram,
    ColorDetectAndDescribe_Image_transformedcolorhistogram,
    ColorDetectAndDescribe_Video_transformedcolorhistogram,
    ColorDetectAndDescribe_Image_colormoments,
    ColorDetectAndDescribe_Video_colormoments,
    ColorDetectAndDescribe_Image_colormomentinvariants,
    ColorDetectAndDescribe_Video_colormomentinvariants,
    ColorDetectAndDescribe_Image_sift,
    ColorDetectAndDescribe_Video_sift,
    ColorDetectAndDescribe_Image_huesift,
    ColorDetectAndDescribe_Video_huesift,
    ColorDetectAndDescribe_Image_hsvsift,
    ColorDetectAndDescribe_Video_hsvsift,
    ColorDetectAndDescribe_Image_opponentsift,
    ColorDetectAndDescribe_Video_opponentsift,
    ColorDetectAndDescribe_Image_rgsift,
    ColorDetectAndDescribe_Video_rgsift,
    ColorDetectAndDescribe_Image_csift,
    ColorDetectAndDescribe_Video_csift,
    ColorDetectAndDescribe_Image_rgbsift,
    ColorDetectAndDescribe_Video_rgbsift,
]
