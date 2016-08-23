"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
LICENSE for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
# pyflann is a toolbox for approximate knn computation
# http://people.cs.ubc.ca/~mariusm/index.php/FLANN/FLANN
# Downloaded version can be founded in './external' directory

import bz2
import gzip
import logging
import os
import os.path as osp
import numpy as np
import csv


# These functions should only be called by colordescriptor.py module, which is
# only available if pyflann is available by the following check
try:
    import pyflann
except ImportError:
    pyflann = None


def flann_load_codebook(filename, is_rowwise=True):
    """
    Load codebook for quantization, and make it ready to be used for pyflann

    To build codebook for colorDescriptor, see the exampleCreateCodebook.py
    script that comes with the colorDescriptor ZIP package. This uses scipy.vq
    kmeans to build the centroid list given a set of colorDescriptor outputs.

    @param filename: file path to the codebook file
    @param is_rowwise: asks if codebook is stored as rowwise. If not, it will be transposed internally after loading.
    @return: row-wise numpy codebook
    """

    cbook = np.loadtxt(filename, delimiter= ',')
    if not is_rowwise:
        cbook = cbook.T

    return cbook


def flann_build(codebook, flann_index_file=None, target_precision=0.99,
                log_level="info"):
    """
    Prepares quantization module based on pyflann

    @param codebook: loaded codebook in numpy format (row-wise)
    @param target_precision: amount of exactness of approximation, the lower,
        the faster
    @param log_level: log data format to be generatede by flann
    """
    log = logging.getLogger(__name__)

    flann = pyflann.FLANN()
    if not flann_index_file or not osp.isfile(flann_index_file):
        flann.build_index(codebook,
                          target_precision=target_precision,
                          log_level=log_level)

        if flann_index_file is not None:
            log.info("Saving generated flann index to: %s",
                     flann_index_file)
            flann.save_index(flann_index_file)
        else:
            log.warning("Constructed new index file "
                        "but did not save it.")

    else:
        log.debug("Given existing index file. Loading it.")
        flann.load_index(flann_index_file, codebook)

    return flann


def flann_quantize_data(flann,
                        filein_name, fileout_name,
                        func_normalize,
                        filein_gzipped=False, filein_bzipped=False,
                        delimiter=' ',
                        fileout_gzipped=False, fileout_bzipped=False,
                        k=10, sparsity=1, size_block=20000):
    """
    Quantize raw data based on flann module

    @param filein_name: input file (raw contents before quantization)
    @param filein_gzipped: is the input file in gziped encoding?
        (default = False)
    @param fileout_name: output file (quantized results)
    @param func_normalize: a function to normalize each data row (if not
        necessary, use None)
    @param delimiter: delimiter used for input file
    @param fileout_gzipped:
    @param k: number of quantization nearest neighbors to be saved (if k>1,
        that's soft quantization)
    @param sparsity: controls the amount of input data to be used. E.g., if 3,
        only every three lines are used.
    @param size_block: number of data points to be used at once. Need to be
        adjusted baesd on memory and CPU IO speed
    """
    assert not (filein_gzipped and filein_bzipped), \
        "Input file cannot be declared as both gzipped and bzipped."
    assert not (fileout_gzipped and fileout_bzipped), \
        "Output file cannot be declared as both gzipped and bzipped."

    fin = None
    if filein_gzipped or os.path.splitext(filein_name)[1] == '.gz':
        fin = gzip.open(filein_name, 'rb')
    elif filein_bzipped or os.path.splitext(filein_name)[1] == '.bz2':
        fin = bz2.BZ2File(filein_name, 'rb')
    else:
        fin = open(filein_name, 'rb')

    count1 = 0  # tracks how many lines are read from file
    count2 = 0  # tracks how many lines are to be processed by being stored in 'lines'
    lines = []

    fout = None
    if fileout_gzipped or os.path.splitext(fileout_name)[1] == '.gz':
        fout = gzip.open(fileout_name, 'w')
    elif fileout_bzipped or os.path.splitext(fileout_name)[1] == '.bz2':
        fout = bz2.BZ2File(fileout_name, 'w')
    else:
        fout = open(fileout_name, 'w')

    def quantize_then_write(lines_in):
        data = np.array(lines_in)
        data = np.float64(data)

        if func_normalize:
            data[:, 3:] = func_normalize(data[:, 3:])

        idx, dists = flann.nn_index(data[:, 3:], k)
        out_data = np.concatenate((data[:, 0:3], idx, dists), axis = 1)
        np.savetxt(fout, out_data, fmt='%g')

    for line in csv.reader(fin, delimiter=' '):
        count1 += 1
        if sparsity != 1 and (count1 % sparsity) == 0:
            continue

        lines.append(line)
        count2 += 1

        if count2 == size_block:
            quantize_then_write(lines)
            count2 = 0 # reset
            lines = []

    if count2 > 0:
        quantize_then_write(lines)
        count2 = 0
        lines = []

    fin.close()
    fout.close()

    return 1


def flann_quantize_data2(flann, in_descriptors, func_normalize=None,
                         k=10, sparsity=1, size_block=20000):
    """
    Quantize raw data based on flan module (in-memory version)

    :param flann: Prepared FLANN module
    :type flann: pyflann.FLANN
    :param in_descriptors: descriptors matrix (colorDescriptor)
    :type in_descriptors: numpy.matrixlib.defmatrix.matrix
    :param func_normalize: a function to normalize each data row (if not
        necessary, use None)
    :param k: number of quantization nearest neighbors to be saved (if k>1,
        that's soft quantization)
    :param sparsity: controls the amount of input data to be used. E.g., if 3,
        only every three lines are used.
    :param size_block: number of data points to be used at once. Need to be
        adjusted baesd on memory and CPU IO speed

    :return:
    :rtype:

    """
    # Input descriptor matrix is of the format:
    # [[ frm, x, y, descriptor... ]
    #  [ ...
    #  ...

    # tracks how many lines are read from file
    count1 = 0

    # Get array base, better suited for loop below
    in_descriptors = in_descriptors.A

    rows = []
    quantized = None

    # Quantization helper function
    # Finds k nearest neighbors to descriptor, creating a vector that is of the
    # format:
    #   [ frm x y nn_1 ... nn_k dist_1 ... dist_k ]
    # which has a total size of (k*2)+3
    # noinspection PyShadowingNames
    def quantize_then_write(rows, quantized):
        data = np.matrix(rows)

        if func_normalize:
            data[:, 3:] = func_normalize(data[:, 3:])

        idx, dists = flann.nn_index(data[:, 3:], k)
        out_data = np.concatenate((data[:, 0:3], idx, dists), axis=1)

        if quantized is None:
            quantized = out_data
        else:
            np.vstack((quantized, out_data))
        return quantized

    # Run through rows in input matrix
    for row in in_descriptors:
        count1 += 1
        if sparsity > 1 and (count1 % sparsity) == 0:
            continue

        rows.append(row)

        if len(rows) == size_block:
            quantized = quantize_then_write(rows, quantized)
            rows = []

    if len(rows) > 0:
        quantized = quantize_then_write(rows, quantized)

    # Output Format:
    # [[ frame, x, y, idx0, ..., idx9, dist0, ..., dist9 ]
    #  [ ... [
    #  ...
    return quantized


def quantizeResults(key, outdir, outtype,
                    pattern_codebook='%s_codebook_med12.txt'):
    # feature file for quantization
    filein_name = os.path.join(outdir, key+'.'+outtype+'-all.txt')
    outname = outtype + '-all.encode.txt'
    fileout_name = os.path.join(outdir, key+'.'+outname)
    # put codebook file under the path
    thisdir = os.path.dirname(os.path.realpath(__file__))
    file_codebook = os.path.join(thisdir, pattern_codebook %outtype)
    # put flann index file under the path
    file_flann = os.path.join(thisdir, '%s.flann' %outtype)
    # load codebook and build flann knn query engine
    cbook = flann_load_codebook(file_codebook, is_rowwise=False)
    flann = flann_build(cbook, file_flann)
    # quantize the data
    flann_quantize_data(flann, filein_name, fileout_name, None)
    flann.delete_index()


def quantizeResults2(file_input, file_output, file_codebook, file_flann,
                     filein_gzipped=False, filein_bzipped=False):
    """
    Another version of raw feature quantization using FLANN.
    No directory structure is assumed. Simply input/ouput/parameter files are
    given.

    @param file_input: Input raw file with first three columns representing
        feature location info
    @param file_output: quantization file to be generated
    @param file_codebook: codebook file
    @param file_flann: quantization parameters for approximate nearest-neighbor
    @param filein_gzipped: if True, treat the file as gzipped version
    """
    cbook = flann_load_codebook(file_codebook, is_rowwise=False)
    flann = flann_build(cbook, file_flann)

    flann_quantize_data(flann, file_input, file_output, None,
                        filein_gzipped=filein_gzipped,
                        filein_bzipped=filein_bzipped)
    flann.delete_index()


def quantizeResults3(in_descriptors, file_codebook, file_flann):
    """
    Another version of raw feature quantization using FLANN.
    No directory structure is assumed. Uses input numpy matrix vs. file.

    :param in_descriptors: Matrix of input descriptors, first row: frame number,
        rows 2-3: colorDescriptor info, rows 4+: descriptor
    :param file_codebook: codebook file
    ;param file_flann: quantization parameters for approximate nearest-neighbor

    """
    cbook = flann_load_codebook(file_codebook, is_rowwise=False)
    flann = flann_build(cbook, file_flann)

    quantized = flann_quantize_data2(flann, in_descriptors)
    flann.delete_index()
    return quantized


#########################################################################
def build_sp_hist(key, outdir, outtype):
    inname  = outtype + '-all.encode.txt'
    filein  = os.path.join(outdir, key+'.'+inname)
    outname = outtype + '-sp.hist.txt'
    fileout = os.path.join(outdir, key+'.'+outname)
    return build_sp_hist_(filein, fileout)


def build_sp_hist_(filein, fileout, bins_code=np.arange(0, 4096 + 1)):
    feas = np.loadtxt(filein)
    cordx = feas[:, 1]
    cordy = feas[:, 2]
    feas = feas[:, 3]  # only the top component, we are looking
    # hard quantization
    # global histogram
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_g = np.histogram(feas, bins=bins_code)[0]
    hist_csift_g = hist_csift_g[np.newaxis]
    # 4 quadrants
    midx = np.ceil(cordx.max()/2)
    midy = np.ceil(cordy.max()/2)
    lx = cordx < midx
    rx = cordx >= midx
    uy = cordy < midy
    dy = cordy >= midy
    # logging.debug("LXUI: %s,%s" % (lx.__repr__(), uy.__repr__()))
    # logging.debug("Length LXUI: %s,%s " % (lx.shape, uy.shape))
    # logging.debug("feas dimensions: %" % (feas.shape,))

    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q1 = np.histogram(feas[lx & uy], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q2 = np.histogram(feas[rx & uy], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q3 = np.histogram(feas[lx & dy], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q4 = np.histogram(feas[rx & dy], bins=bins_code)[0]
    hist_csift_q1 = hist_csift_q1[np.newaxis]
    hist_csift_q2 = hist_csift_q2[np.newaxis]
    hist_csift_q3 = hist_csift_q3[np.newaxis]
    hist_csift_q4 = hist_csift_q4[np.newaxis]

    # 3 layers
    ythird = np.ceil(cordy.max()/3)
    l1 = cordy <= ythird
    l2 = (cordy > ythird) & (cordy <= 2*ythird)
    l3 = cordy > 2*ythird
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_l1 = np.histogram(feas[l1], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_l2 = np.histogram(feas[l2], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_l3 = np.histogram(feas[l3], bins=bins_code)[0]
    hist_csift_l1 = hist_csift_l1[np.newaxis]
    hist_csift_l2 = hist_csift_l2[np.newaxis]
    hist_csift_l3 = hist_csift_l3[np.newaxis]
    # concatenate
    hist_csift = np.vstack((hist_csift_g, hist_csift_q1, hist_csift_q2,
                            hist_csift_q3, hist_csift_q4, hist_csift_l1,
                            hist_csift_l2, hist_csift_l3))
    np.savetxt(fileout, hist_csift, fmt='%g')
    return hist_csift


def build_sp_hist2(feas, bins_code=np.arange(0, 4096+1)):
    """ Build spacial pyramid from quantized data

    :param feas: quantized data matrix
    :type feas: numpy.matrixlib.defmatrix.matrix

    :return: martrix of 8 rows representing the histograms for the different
        spacial regions.
    :rtype: numpy.matrixlib.defmatrix.matrix

    """
    cordx = feas[:, 1]
    cordy = feas[:, 2]
    feas = feas[:, 3]  # only the top component, we are looking
    # hard quantization
    # global histogram
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_g = np.histogram(feas, bins=bins_code)[0]
    hist_csift_g = hist_csift_g[np.newaxis]
    # 4 quadrants
    # noinspection PyTypeChecker
    midx = np.ceil(cordx.max()/2)
    # noinspection PyTypeChecker
    midy = np.ceil(cordy.max()/2)
    lx = cordx < midx
    rx = cordx >= midx
    uy = cordy < midy
    dy = cordy >= midy
    # logging.error("LXUI: %s,%s", lx.__repr__(), uy.__repr__())
    # logging.error("Length LXUI: %s,%s", lx.shape, uy.shape)
    # logging.error("feas dimensions: %s", feas.shape)

    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q1 = np.histogram(feas[lx & uy], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q2 = np.histogram(feas[rx & uy], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q3 = np.histogram(feas[lx & dy], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_q4 = np.histogram(feas[rx & dy], bins=bins_code)[0]
    hist_csift_q1 = hist_csift_q1[np.newaxis]
    hist_csift_q2 = hist_csift_q2[np.newaxis]
    hist_csift_q3 = hist_csift_q3[np.newaxis]
    hist_csift_q4 = hist_csift_q4[np.newaxis]

    # 3 layers
    # noinspection PyTypeChecker
    ythird = np.ceil(cordy.max()/3)
    l1 = cordy <= ythird
    l2 = (cordy > ythird) & (cordy <= 2*ythird)
    l3 = cordy > 2*ythird
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_l1 = np.histogram(feas[l1], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_l2 = np.histogram(feas[l2], bins=bins_code)[0]
    #: :type: numpy.core.multiarray.ndarray
    hist_csift_l3 = np.histogram(feas[l3], bins=bins_code)[0]
    hist_csift_l1 = hist_csift_l1[np.newaxis]
    hist_csift_l2 = hist_csift_l2[np.newaxis]
    hist_csift_l3 = hist_csift_l3[np.newaxis]
    # concatenate
    hist_csift = np.vstack((hist_csift_g, hist_csift_q1, hist_csift_q2,
                            hist_csift_q3, hist_csift_q4, hist_csift_l1,
                            hist_csift_l2, hist_csift_l3))
    return hist_csift
