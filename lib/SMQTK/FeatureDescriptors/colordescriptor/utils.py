"""
Utility functions for colorDescriptor operations
"""

import logging
import numpy
import os
import os.path as osp
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
        raise ImportError("Cannot find the DescriptorIO module provided by "
                          "ColorDescriptor. Read the README for dependencies!")


def generate_descriptors(cd_exe, img_filepath, descriptor_type,
                         info_checkpoint_path=None,
                         descr_checkpoint_path=None):
    """
    Execute the given colorDescriptor executable, returning the generated info
    and descriptor matrices for the provided image. Descriptor matrix is
    normalized into histograms of relative frequencies instead of histograms of
    bin counts.

    If either checkpoint file paths are not provided, we will always compute the
    descriptors for the given image file using the provided executable. If both
    checkpoint files exist, we load their contents and return their contents as
    is. If both checkpoint filepaths were provided and we generated new content
    with the provided executable, the results are saved to their respective
    checkpoint files. The normalized descriptor matrix is saved.

    :param cd_exe: ColorDescriptor executable to use
    :type cd_exe: str

    :param img_filepath: Path to the image file to process
    :type img_filepath: str

    :param descriptor_type: String type of descriptor to use from
        colorDescriptor.
    :type descriptor_type: str

    :param info_checkpoint_path: Path to where the computed information matrix
        for the given file should be.
    :type info_checkpoint_path: str

    :param descr_checkpoint_path: Path to where the computed descriptor matrix
        for the given file should be.
    :type descr_checkpoint_path: str

    :return: 2 matrices: the metadata matrix and normalized descriptor matrix
        (relative frequency histograms).
    :rtype: (numpy.core.multiarray.ndarray, numpy.core.multiarray.ndarray)

    """
    log = logging.getLogger("ColorDescriptor::generate_descriptors{%s,%s}"
                            % (descriptor_type, osp.basename(img_filepath)))

    if info_checkpoint_path and descr_checkpoint_path \
            and osp.isfile(info_checkpoint_path) \
            and osp.isfile(descr_checkpoint_path):
        log.debug("Found existing checkpoint files, loading those.")
        i = numpy.load(info_checkpoint_path)
        d = numpy.load(descr_checkpoint_path)
        return i, d

    # Determine the spacing between sample points in the image. We want have at
    # least 50 sample points along the shortest side with a minimum of 6 pixels
    # distance between sample points.
    w, h = PIL.Image.open(img_filepath).size
    ds_spacing = max(int(min(w, h) / 50.0), 6)
    log.debug("dense-sample spacing: %d", ds_spacing)

    tmp_fd, tmp_path = tempfile.mkstemp()
    os.close(tmp_fd)

    log.debug("launching executable subprocess")
    cmd = [cd_exe, img_filepath, '--output', tmp_path,
           '--detector', 'densesampling',
           '--ds_spacing', str(ds_spacing),
           '--descriptor', descriptor_type]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()

    log.debug("normalizing histogram into relative frequency")
    # Info matrix consists of [x, y, scale, orientation, corner-ness]
    # - See colorDescriptor documentation for more information
    info, descriptors = DescriptorIO.readDescriptors(tmp_path)

    # Divides each row in the descriptors matrix with the row-wise sum.
    # - This results in histograms for relative frequencies instead of direct
    #   bin counts.
    # - Adding float_info.min to row sums to prevent div-by-zero exception while
    #   introducing minimal numerical error.
    # noinspection PyUnresolvedReferences
    descriptors = descriptors / (numpy.matrix(descriptors).sum(axis=1) +
                                 sys.float_info.min).A

    if info_checkpoint_path and descr_checkpoint_path:
        numpy.save(info_checkpoint_path, info)
        numpy.save(descr_checkpoint_path, descriptors)

    return info, descriptors
