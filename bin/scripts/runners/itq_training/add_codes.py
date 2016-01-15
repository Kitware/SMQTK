import logging
import json
import multiprocessing
import re
import time

import numpy as np

from smqtk.representation import DescriptorElementFactory
from smqtk.utils import SmqtkObject
from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.bit_utils import bit_vector_to_int
from smqtk.utils.jsmin import jsmin

from load_algo import load_algo


UUIDS_FILEPATH = "descriptor_uuids.all.txt"
ITQ_ROTATION = "itq.256.rotation.npy"
ITQ_MEAN_VEC = "itq.256.mean_vec.npy"


fn_sha1_re = re.compile("\w+\.(\w+)\.vector\.npy")

element_type_str = open('descriptor_type_name.txt').read().strip()

factory_config = json.loads(jsmin(open("descriptor_factory_config.json").read()))
factory = DescriptorElementFactory.from_config(factory_config)


#
# Multiprocessing of ITQ small-code generation
#
def make_element(uuid):
    return factory.new_descriptor(element_type_str, uuid)


def make_elements_from_uuids(uuids):
    for uuid in uuids:
        yield make_element(uuid)


class SmallCodeProcess (SmqtkObject, multiprocessing.Process):
    """
    Worker process for ITQ smallcode generation given a rotation matrix and mean vector.

    Input queue format: PostgresDescriptorElement
    Output queue format: (int|long, PostgresDescriptorElement)

    Terminal value: None

    """

    def __init__(self, i, in_q, out_q, r, mean_vec, batch=500):
        super(SmallCodeProcess, self).__init__()
        self.i = i
        self.in_q = in_q
        self.out_q = out_q
        self.r = r
        self.m_vec = mean_vec
        self.batch = batch

    def run(self):
        self._log.debug("[%s] Starting worker", self.name)

        packet = self.in_q.get()
        d_elems = []
        while packet:
            # self._log.debug("[%s] Packet: %s", self.name, packet)
            descr_elem = packet

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


def async_compute_smallcodes(r, mean_vec, descr_elements,
                             procs=None, report_interval=1.):
    """
    Returns tuples of small-code values with the associated DescriptorElement
    instance.
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

    log.info("Loading descriptor UUIDs")
    with open(UUIDS_FILEPATH) as f:
        descriptor_uuids = [l.strip() for l in f]

    log.info("Loading ITQ components")
    r = np.load(ITQ_ROTATION)
    mv = np.load(ITQ_MEAN_VEC)

    log.info("Making small-codes")
    sc_d_pairs = async_compute_smallcodes(
        r, mv, make_elements_from_uuids(descriptor_uuids)
    )

    log.info("Loading ITQ model")
    itq_index = load_algo()

    log.info("Adding small codes")
    itq_index._code_index.add_many_descriptors(sc_d_pairs)

    return descriptor_uuids, itq_index


if __name__ == "__main__":
    initialize_logging(logging.getLogger(), logging.DEBUG)
    uuids, itq_index = add_descriptors_smallcodes()
