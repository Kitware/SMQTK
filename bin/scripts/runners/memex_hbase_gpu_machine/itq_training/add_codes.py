import cPickle
import logging
import multiprocessing
import re
import time

import numpy as np

from smqtk.algorithms.nn_index.lsh.itq import ITQNearestNeighborsIndex
from smqtk.representation.descriptor_element.local_elements import DescriptorFileElement
from smqtk.utils import SmqtkObject
from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.bit_utils import bit_vector_to_int

from load_algo import load_algo


DESCRIPTORS_ROOT_DIR = "/data/kitware/smqtk/image_cache_cnn_compute/descriptors"
DESCRIPTORS_FILE_NAMES = "/data/kitware/smqtk/image_cache_cnn_compute/descriptor_file_names.5.3mil.pickle"
ITQ_ROTATION = "/data/kitware/smqtk/image_cache_cnn_compute/itq_model/256-bit/rotation.npy"
ITQ_MEAN_VEC = "/data/kitware/smqtk/image_cache_cnn_compute/itq_model/256-bit/mean_vec.npy"


fn_sha1_re = re.compile("\w+\.(\w+)\.vector\.npy")


#
# Multiprocessing of ITQ small-code generation
#
def make_element(sha1):
    return DescriptorFileElement("CaffeDefaultImageNet", sha1,
                                 DESCRIPTORS_ROOT_DIR,
                                 10)


def make_elements_from_filenames(filenames):
    for fn in filenames:
        yield make_element(fn_sha1_re.match(fn).group(1))


class SmallCodeProcess (SmqtkObject, multiprocessing.Process):
    """
    Worker process for ITQ smallcode generation given a rotation matrix and mean vector.

    Input queue format: DescriptorFileElement
    Output queue format: (int|long, DescriptorFileElement)

    Terminal value: None

    """

    # class ItqShell (ITQNearestNeighborsIndex):
    #     """
    #     Shell subclass for access to small-code calculation method
    #     """
    #     def __init__(self, rot, mean_vec):
    #         super(SmallCodeProcess.ItqShell, self).__init__(code_index=None)
    #         self._r = rot
    #         self._mean_vector = mean_vec

    def __init__(self, i, in_q, out_q, r, mean_vec, batch=500):
        super(SmallCodeProcess, self).__init__()
        self._log.debug("[%s] Starting worker", self.name)
        self.in_q = in_q
        self.out_q = out_q
        self.r = r
        self.m_vec = mean_vec
        self.batch = batch

    def run(self):
        # shell = self.ItqShell(self.r, self.m_vec)

        packet = self.in_q.get()
        d_elems = []
        while packet:
            # self._log.debug("[%s] Packet: %s", self.name, packet)
            descr_elem = packet
            # self.out_q.put((shell.get_small_code(descr_elem), 
            #                 descr_elem))

            d_elems.append(descr_elem)
            if len(d_elems) >= self.batch:
                self._log.debug("[%s] Computing batch of %d", self.name, len(d_elems))
                m = np.array([d.vector() for d in d_elems])
                z = np.dot((m - self.m_vec), self.r)
                b = np.zeros(z.shape, dtype=np.uint8)
                b[z >= 0] = 1
                for bits, d in zip(b, d_elems):
                    self.out_q.put((bit_vector_to_int(bits), d))
                d_elems = []

            packet = self.in_q.get()

        if d_elems:
            self._log.debug("[%s] Computing batch of %d", self.name, len(d_elems))
            m = np.array([d.vector() for d in d_elems])
            z = np.dot((m - self.m_vec), self.r)
            b = np.zeros(z.shape, dtype=np.uint8)
            b[z >= 0] = 1
            for bits, d in zip(b, d_elems):
                self.out_q.put((bit_vector_to_int(bits), d))
            d_elems = []


def async_compute_smallcodes(r, mean_vec, descr_elements,
                             procs=None, report_interval=1.):
    """
    Yields (int|long, DescriptorElement)
    """
    log = logging.getLogger(__name__)

    if procs is None:
        procs = multiprocessing.cpu_count()

    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue(procs*2)

    workers = [SmallCodeProcess(i, in_q, out_q, r, mean_vec) for i in range(procs)]
    for w in workers:
        w.daemon = True

    sc_d_return = []
    try:
        log.info("Starting worker processes")
        for w in workers:
            w.start()

        log.info("Sending elements")
        s = 0
        lt = t = time.time()
        for de in descr_elements:
            in_q.put(de)
            
            s += 1
            if time.time() - lt >= report_interval:
                log.debug("Sent packets per second: %f, Total: %d",
                    s / (time.time() - t), s
                )
                lt = time.time()
        # Send terminal packets at tail
        for w in workers:
            in_q.put(None)
        in_q.close()

        log.info("Collecting small codes")
        r = 0
        lt = t = time.time()
        for i in xrange(s):
            sc, d = out_q.get()
            sc_d_return.append((sc, d))

            r += 1
            if time.time() - lt >= report_interval:
                log.debug("Collected packets per second: %f, Total: %d",
                    r / (time.time() - t), r
                )
                lt = time.time()
        out_q.close()
        log.info("Scanned all smallcodes")

        return sc_d_return
        
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
            w.join()
        for q in (in_q, out_q):
            q.close()
            q.join_thread()


def add_descriptors_smallcodes():
    log = logging.getLogger(__name__)

    log.info("Loading descriptor file names")
    with open(DESCRIPTORS_FILE_NAMES) as f:
        descriptor_filenames = cPickle.load(f)
    log.info("Loading ITQ components")
    r = np.load(ITQ_ROTATION)
    mv = np.load(ITQ_MEAN_VEC)

    log.info("Making SC iterator")
    sc_d_pairs = async_compute_smallcodes(
        r, mv, make_elements_from_filenames(descriptor_filenames)
    )

    log.info("Loading ITQ model")
    itq_index = load_algo()

    log.info("Adding small codes")
    itq_index._code_index.add_many_descriptors(sc_d_pairs)

    return descriptor_filenames, itq_index


if __name__ == "__main__":
    initialize_logging(logging.getLogger(), logging.DEBUG)
    filenames, itq_index = add_descriptors_smallcodes()
