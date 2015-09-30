import logging
import multiprocessing
import multiprocessing.pool
import os.path as osp
import shutil
import uuid

from smqtk.algorithms.relevancy_index import get_relevancy_index_impls
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
from smqtk.representation.descriptor_index.memory import DescriptorMemoryIndex
from smqtk.utils import plugin
from smqtk.utils import safe_create_dir


DFLT_MEMORY_DESCR_FACTORY = DescriptorElementFactory(DescriptorMemoryElement,
                                                     {})
DFLT_REL_INDEX_CONFIG = {
    "type": "LibSvmHikRelevancyIndex",
    "LibSvmHikRelevancyIndex": {
        "descr_cache_filepath": None,
    }
}


class IqrResultsDict (dict):
    """
    Dictionary subclass for containing DescriptorElement-to-float mapping.

    We expect keys to be DescriptorElement instances and the values to be floats
    between [0,1], inclusive.

    """

    def __setitem__(self, i, v):
        super(IqrResultsDict, self).__setitem__(i, float(v))

    def update(self, other=None, **kwds):
        """
        D.update([E, ]**F) -> None. Update D from dict/iterable E and F.
        If E present and has a .keys() method, does: for k in E: D[k] = E[k]
        If E present and lacks .keys() method, does: for (k, v) in E: D[k] = v
        In either case, this is followed by: for k in F: D[k] = F[k]

        Reimplemented so as to use override __setitem__ method.
        """
        if hasattr(other, 'keys'):
            for k in other:
                self[k] = float(other[k])
        elif other is not None:
            for k, v in other:
                self[k] = float(v)
        for k in kwds:
            self[k] = float(kwds[k])


class IqrSession (object):
    """
    Encapsulation of IQR Session related data structures with a centralized lock
    for multi-thread access.

    This object is compatible with the python with-statement, so when elements
    are to be used or modified, it should be within a with-block so race
    conditions do not occur across threads/sub-processes.

    """

    def __init__(self, work_directory, descriptor, nn_index,
                 pos_seed_neighbors=500,
                 rel_index_config=DFLT_MEMORY_DESCR_FACTORY,
                 descriptor_factory=DFLT_MEMORY_DESCR_FACTORY,
                 session_uid=None):
        """ Initialize the IQR session

        This does not initialize the working index for ranking as there are no
        known positive descriptor examples at this time.

        Adjudications
        -------------
        Adjudications are carried through between initializations. This allows
        indexed material adjudicated through-out the lifetime of the session to
        stay relevant.

        :param work_directory: Directory assigned to this session for temporary
            and working files.
        :type work_directory: str

        :param descriptor: Descriptor to use for this IQR session
        :type descriptor:
            smqtk.algorithms.descriptor_generator.DescriptorGenerator

        :param nn_index: NearestNeighborIndex to draw from when initializing IQR
            session.
        :type nn_index: smqtk.algorithms.nn_index.NearestNeighborsIndex

        :param pos_seed_neighbors: Number of neighbors to pull from the given
            ``nn_index`` for each positive exemplar when populating the working
            index, i.e. this value determines the size of the working index for
            IQR refinement. By default, we try to get 500 neighbors.

            Since there may be partial to significant overlap of near neighbors
            as a result of nn_index queries for positive exemplars, the working
            index may contain anywhere from this value's number of entries, to
            ``N*P``, where ``N`` is this value and ``P`` is the number of
            positive examples at the time of working index initialization.
        :type pos_seed_neighbors: int

        :param rel_index_config: Plugin configuration dictionary for the
            RelevancyIndex to use for ranking user adjudications. By default we
            we use an in-memory libSVM based index using the histogram
            intersection metric.
        :type rel_index_config: dict

        :param descriptor_factory: DescriptorElementFactory instance to use to
            produce new descriptors in output extension data. By default, we
            use a factory that produces in-memory descriptors.
        :type descriptor_factory: DescriptorElementFactory

        :param session_uid: Optional manual specification of session UUID.
        :type session_uid: str or uuid.UUID

        """
        self.uuid = session_uid or uuid.uuid1()
        self.lock = multiprocessing.RLock()

        # Local descriptor index for ranking, populated by a query to the
        #   nn_index instance.
        # Added external data/descriptors not added to this index.
        self.working_index = DescriptorMemoryIndex()

        # Descriptor references from our index (above) that have been
        #   adjudicated.
        #: :type: set[smqtk.representation.DescriptorElement]
        self.positive_descriptors = set()
        #: :type: set[smqtk.representation.DescriptorElement]
        self.negative_descriptors = set()

        # Example pos/neg data/descriptors added to this session
        # (external to our working index).
        #: :type: dict[smqtk.representation.DataElement, smqtk.representation.DescriptorElement]
        self.ex_pos_data2descriptor = dict()
        #: :type: dict[smqtk.representation.DataElement, smqtk.representation.DescriptorElement]
        self.ex_neg_data2descriptor = dict()

        self.pos_seed_neighbors = int(pos_seed_neighbors)

        # Working directory assigned to this session
        self._work_dir = work_directory

        # Mapping of a DescriptorElement in our relevancy search index (not the
        #   index that the nn_index uses) to the relevancy score given the
        #   recorded positive and negative adjudications.
        # This is None before any initialization or refinement occurs.
        #: :type: None or dict of (collections.Hashable, float)
        self.results = None

        #
        # Algorithm Instances [+Config]
        #
        # DescriptorGenerator instance assigned to this session.
        self.descriptor = descriptor
        # Factory for generating DescriptorElements of a configured impl type.
        self.descriptor_factory = descriptor_factory
        # NearestNeighborIndex instance assigned to this session.
        self.nn_index = nn_index
        # RelevancyIndex configuration and instance that is used for producing
        #   results.
        # This is only [re]constructed when initializing the session.
        self.rel_index_config = rel_index_config
        # This is None until session initialization happens after pos/neg
        # exemplar data has been added.
        #: :type: None | smqtk.algorithms.relevancy_index.RelevancyIndex
        self.rel_index = None

    def __del__(self):
        # Clean up working directory
        if osp.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)

    def __enter__(self):
        """
        :rtype: IqrSession
        """
        self.lock.acquire()
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    @property
    def _log(self):
        return logging.getLogger(
            '.'.join((self.__module__, self.__class__.__name__)) +
            "[%s]" % self.uuid
        )

    @property
    def work_dir(self):
        safe_create_dir(self._work_dir)
        return self._work_dir

    def ordered_results(self):
        """
        Return a tuple of the current (id, probability) result pairs in
        order of probability score. If there are no results yet, None is
        returned.

        :rtype: None | tuple[(smqtk.representation.DescriptorElement, float)]

        """
        with self.lock:
            if self.results:
                return tuple(sorted(self.results.iteritems(),
                                    key=lambda p: p[1],
                                    reverse=True))
            return None

    def add_positive_data(self, *data_elements):
        """
        Add one or more data elements to this IQR session as positive examples.
        This produces descriptors on the input data with our configured
        descriptor generator.

        :param data_elements: Iterable of data elements to add as positive
            examples.
        :type data_elements: collections.Iterable[smqtk.representation.DataElement]

        """
        with self.lock:
            self.ex_pos_data2descriptor.update(
                self.descriptor.compute_descriptor_async(
                    data_elements, self.descriptor_factory
                )
            )

    def add_negative_data(self, *data_elements):
        """
        Add one or more data elements to this IQR session as negative examples.
        This produces descriptors on the input data with our configured
        descriptor generator.

        :param data_elements: Iterable of data elements to add as positive
            examples.
        :type data_elements: collections.Iterable[smqtk.representation.DataElement]

        """
        with self.lock:
            self.ex_neg_data2descriptor.update(
                self.descriptor.compute_descriptor_async(
                    data_elements, self.descriptor_factory
                )
            )

    def initialize(self):
        """
        Initialize working index based on currently set positive exemplar data.

        This takes into account the currently set positive data descriptors as
        well as positively adjudicated descriptors from the lifetime of this
        session.

        :raises RuntimeError: There are no positive example descriptors in this
            session to use as a basis for querying.

        """
        if len(self.ex_pos_data2descriptor) + \
                len(self.positive_descriptors) <= 0:
            raise RuntimeError("No positive descriptors to query the neighbor "
                               "index with.")
        # Clear the current working index so we can put different things in it
        self._log.info("Clearing working index")
        self.working_index.clear()

        # build up new working index
        for p in self.ex_pos_data2descriptor.itervalues():
            self._log.info("Querying neighbors to: %s", p)
            self.working_index.add_many_descriptors(
                self.nn_index.nn(p, n=self.pos_seed_neighbors)[0]
            )
        for p in self.positive_descriptors:
            self._log.info("Querying neighbors to: %s", p)
            self.working_index.add_many_descriptors(
                self.nn_index.nn(p, n=self.pos_seed_neighbors)[0]
            )

        # Make new relevancy index
        self._log.info("Creating new relevancy index over working index.")
        #: :type: smqtk.algorithms.relevancy_index.RelevancyIndex
        self.rel_index = plugin.from_plugin_config(self.rel_index_config,
                                                   get_relevancy_index_impls)
        self.rel_index.build_index(self.working_index.iterdescriptors())

    def adjudicate(self, new_positives=(), new_negatives=(),
                   un_positives=(), un_negatives=()):
        """
        Update current state of working index positive and negative
        adjudications based on descriptor UUIDs.

        :param new_positives: Descriptors of elements in our working index to
            now be considered to be positively relevant.
        :type new_positives: collections.Iterable[smqtk.representation.DescriptorElement]

        :param new_negatives: Descriptors of elements in our working index to
            now be considered to be negatively relevant.
        :type new_negatives: collections.Iterable[smqtk.representation.DescriptorElement]

        :param un_positives: Descriptors of elements in our working index to now
            be considered not positive any more.
        :type un_positives: collections.Iterable[smqtk.representation.DescriptorElement]

        :param un_negatives: Descriptors of elements in our working index to now
            be considered not negative any more.
        :type un_negatives: collections.Iterable[smqtk.representation.DescriptorElement]

        """
        with self.lock:
            self.positive_descriptors.update(new_positives)
            self.positive_descriptors.difference_update(un_positives)
            self.positive_descriptors.difference_update(new_negatives)

            self.negative_descriptors.update(new_negatives)
            self.negative_descriptors.difference_update(un_negatives)
            self.negative_descriptors.difference_update(new_positives)

    def refine(self):
        """ Refine current model results based on current adjudication state

        :raises RuntimeError: There are no adjudications to run on. We must have
            at least one positive adjudication.

        """
        with self.lock:
            if not self.positive_descriptors:
                raise RuntimeError("Did not find at least one positive "
                                   "adjudication.")
            if not self.rel_index:
                raise RuntimeError("No relevancy index yet. Must not have "
                                   "initialize session (no working index).")

            # fuse pos/neg adjudications + added positive data descriptors
            pos = self.ex_pos_data2descriptor.values() + \
                list(self.positive_descriptors)
            neg = self.ex_neg_data2descriptor.values() + \
                list(self.negative_descriptors)
            id_probability_map = self.rel_index.rank(pos, neg)

            if self.results is None:
                self.results = IqrResultsDict()
            self.results.update(id_probability_map)

            # # Force adjudicated positives and negatives to be probability 1 and
            # # 0, respectively, since we want to control where they show up in
            # # our results view.
            # # - Not all pos/neg descriptors may be in our working index.
            # for d in pos:
            #     if d in self.results:
            #         self.results[d] = 1.0
            # for d in neg:
            #     if d in self.results:
            #         self.results[d] = 0.0

    def reset(self):
        """ Reset the IQR Search state

        No positive adjudications, reload original feature data

        """
        with self.lock:
            self.working_index.clear()
            self.positive_descriptors.clear()
            self.negative_descriptors.clear()
            self.ex_pos_data2descriptor.clear()
            self.ex_neg_data2descriptor.clear()

            self.rel_index = None
            self.results = None

            # clear contents of working directory
            shutil.rmtree(self.work_dir)
