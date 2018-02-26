"""
Utility functions for colorDescriptor operations
"""

import logging
import numpy
import os
import os.path as osp
# noinspection PyPackageRequirements
import PIL.Image
import subprocess
import sys
import tempfile

# Try to import DescriptorIO module provided by ColorDescriptor package
try:
    import DescriptorIO
except ImportError:
    try:
        from colorDescriptor import DescriptorIO
    except ImportError:
        DescriptorIO = None


def has_colordescriptor_module():
    """
    :return: Boolean describing whether the required colorDescriptor module was
        found or not. If not found, the base descriptor functionality does not
        exist.
    :rtype: bool
    """
    return DescriptorIO is not None


def generate_descriptors(cd_exe, img_filepath, descriptor_type,
                         info_matrix_path, descr_matrix_path,
                         limit_descriptors=None,
                         recompute=False):
    """
    Execute the given colorDescriptor executable, saving the generated info
    and descriptor matrices for the provided image to the provided file paths.
    Descriptor matrix is normalized into histograms of relative frequencies
    instead of histograms of raw bin counts.

    This does NOT return matrices directly due to memory concerns, especially
    in regards to multiprocessing as multiple copies of the matrix exist in the
    system, leading to excessive memory clogging.

    Matrices are saved in numpy binary format (.npy). ``numpy.load`` function
    should be used to load matrices back in.

    :raises ImportError: The required python module for colorDescriptor IO is
        not available.
    :raises RuntimeError: Failed to generate output files or matrices for the '
        given input.

    :param cd_exe: ColorDescriptor executable to use
    :type cd_exe: str

    :param img_filepath: Path to the image file to process
    :type img_filepath: str

    :param descriptor_type: String type of descriptor to use from
        colorDescriptor.
    :type descriptor_type: str

    :param info_matrix_path: Path to where the computed information matrix
        for the given file should be saved. This will be saved as a numpy
        binary file (.npy).
    :type info_matrix_path: str

    :param descr_matrix_path: Path to where the computed descriptor matrix
        for the given file should be saved. This will be saved as a numpy
        binary file (.npy).
    :type descr_matrix_path: str

    :param limit_descriptors: Limit the number of descriptors generated if we
        were to produce more than the limit. If we exceed the limit, we randomly
        subsample down to the limit.
    :type limit_descriptors: int

    :param recompute: Force re-computation of descriptors for the given image
        file. This causes possible existing output files to be overwritten.
    :type recompute: bool

    :return: Shape information for info and descriptor matrices
    :rtype: ((int, int), (int, int))

    """
    if not has_colordescriptor_module():
        raise ImportError("Cannot find the DescriptorIO module provided by "
                          "ColorDescriptor. Read the README for dependencies!")

    log = logging.getLogger("ColorDescriptor::generate_descriptors{%s,%s}"
                            % (descriptor_type, osp.basename(img_filepath)))

    if not recompute \
            and osp.isfile(info_matrix_path) \
            and osp.isfile(descr_matrix_path):
        # log.debug("Found existing matrix files, loading shapes.")
        return (numpy.load(info_matrix_path).shape,
                numpy.load(descr_matrix_path).shape)

    # Determine the spacing between sample points in the image. We want have at
    # least 50 sample points along the shortest side with a minimum of 6 pixels
    # distance between sample points.
    try:
        w, h = PIL.Image.open(img_filepath).size
    except IOError as ex:
        raise RuntimeError("Could not open image at filepath '%s': %s"
                           % (img_filepath, str(ex)))
    ds_spacing = max(int(min(w, h) / 50.0), 6)
    log.debug("dense-sample spacing: %d", ds_spacing)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix='colorDescriptor.')
    os.close(tmp_fd)

    log.debug("launching executable subprocess")
    # TODO: Perform harrislaplace detection method, if yields 0 descriptors, run
    #       densesample method. When harrislaplace fails, this generally means
    #       that there are no edge features in the image, e.g. a solid color
    #       image.
    cmd = [cd_exe, img_filepath, '--output', tmp_path,
           # "harrislaplace" is another option here, but this can result in 0
           # descriptors for an image, and is also slower.
           '--detector', 'densesampling',
           # '--detector', 'harrislaplace',
           '--ds_spacing', str(ds_spacing),
           '--descriptor', descriptor_type]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()

    # Info matrix consists of [x, y, scale, orientation, corner-ness]
    # - See colorDescriptor documentation for more information
    try:
        log.debug("Reading descriptors output")
        info, descriptors = DescriptorIO.readDescriptors(tmp_path)
    except IOError as ex:
        raise RuntimeError("ColorDescriptor failed to generate proper output "
                           "file. See error log for details. (error: %s"
                           % str(ex))
    finally:
        os.remove(tmp_path)

    # Also error if the descriptor is empty
    if not descriptors.shape[1]:
        raise RuntimeError("Produced empty descriptor.")

    # Divides each row in the descriptors matrix with the row-wise sum.
    # - This results in histograms for relative frequencies instead of direct
    #   bin counts.
    # - Adding float_info.min to row sums to prevent div-by-zero exception while
    #   introducing minimal numerical error.
    # noinspection PyUnresolvedReferences
    log.debug("normalizing histogram into relative frequency")
    # noinspection PyUnresolvedReferences
    descriptors = descriptors / (numpy.matrix(descriptors).sum(axis=1) +
                                 sys.float_info.min).A

    # Randomly sample rows down to this count if what was generated exceeded the
    # limit.
    if limit_descriptors and info.shape[0] > limit_descriptors:
        idxs = numpy.random.permutation(numpy.arange(info.shape[0]))[:limit_descriptors]
        idxs = sorted(idxs)
        info = info[idxs, :]
        descriptors = descriptors[idxs, :]

    numpy.save(info_matrix_path, info)
    numpy.save(descr_matrix_path, descriptors)
    return info.shape, descriptors.shape
