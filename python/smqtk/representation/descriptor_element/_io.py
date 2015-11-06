import logging
import multiprocessing

import numpy

from smqtk.utils import SmqtkObject


__author__ = 'paul.tunison@kitware.com'


__all__ = [
    'elements_to_matrix',
]


def elements_to_matrix(descr_elements, mat=None, procs=None):
    """
    Add to or create a numpy matrix, adding to it the vector data contained in
    a sequence of DescriptorElement instances using asynchronous processing.

    If ``mat`` is provided, its shape must equal:
        ( len(descr_elements) , descr_elements[0].size )

    :param descr_elements:
    :param mat:
    :param procs:

    :return: Create or input matrix


    """
    log = logging.getLogger(__name__)

    # Special case for in-memory storage of descriptors
    from smqtk.representation.descriptor_element.local_elements \
        import DescriptorMemoryElement

    # Create/check matrix
    if mat is None:
        mat = numpy.ndarray((len(descr_elements),
                             descr_elements[0].vector().size),
                            float)
    else:
        assert mat.shape[0] == len(descr_elements)
        assert mat.shape[1] == descr_elements[0].vector().size

    if procs is None:
        procs = multiprocessing.cpu_count()

    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue(procs * 2)

    # Workers for async extraction
    workers = [_ElemVectorExtractor(in_q, out_q) for _ in xrange(procs)]

    async_packets_sent = 0
    for r, d in enumerate(descr_elements):
        if isinstance(d, DescriptorMemoryElement):
            # No loading required, already in memory
            mat[r] = d.vector()
        else:
            in_q.put((r, d))
            async_packets_sent += 1

    # Collect work from async
    for _ in xrange(async_packets_sent):
        r, v = out_q.get()
        mat[r] = v
    out_q.close()

    # All work should be exhausted at this point
    assert in_q.qsize() == 0
    assert out_q.qsize() == 0

    # Shutdown workers
    # - Workers should exit upon getting a None packet
    for _ in workers:
        # One for each worker
        in_q.put(None)
    in_q.close()
    for w in workers:
        w.join()

    return mat


class _ElemVectorExtractor (SmqtkObject, multiprocessing.Process):
    """
    Helper process for extracting DescriptorElement vectors on a separate
    process. This terminates with a None packet fed to in_q. Otherwise, in_q
    values are expected to be (row, element) pairs. Tuples of the form
    (row, vector) are published to the out_q.

    Terminal value: None

    """

    def __init__(self, in_q, out_q):
        super(_ElemVectorExtractor, self)\
            .__init__()
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        packet = self.in_q.get()
        while packet is not None:
            row, elem = packet
            self.out_q.put((row, elem.vector()))
            packet = self.in_q.get()
