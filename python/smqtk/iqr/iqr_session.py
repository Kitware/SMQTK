import io
import json
import logging
import threading
import uuid
import zipfile

import six

from smqtk.algorithms.relevancy_index import get_relevancy_index_impls
from smqtk.representation.descriptor_index.memory import MemoryDescriptorIndex
from smqtk.utils import SmqtkObject
from smqtk.utils import plugin


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

        Reimplemented so as to use override __setitem__ method so values are
        known to be floats.
        """
        if hasattr(other, 'keys'):
            for k in other:
                self[k] = float(other[k])
        elif other is not None:
            for k, v in other:
                self[k] = float(v)
        for k in kwds:
            self[k] = float(kwds[k])


class IqrSession (SmqtkObject):
    """
    Encapsulation of IQR Session related data structures with a centralized lock
    for multi-thread access.

    This object is compatible with the python with-statement, so when elements
    are to be used or modified, it should be within a with-block so race
    conditions do not occur across threads/sub-processes.

    """

    @property
    def _log(self):
        return logging.getLogger(
            '.'.join((self.__module__, self.__class__.__name__)) +
            "[%s]" % self.uuid
        )

    def __init__(self, pos_seed_neighbors=500,
                 rel_index_config=DFLT_REL_INDEX_CONFIG,
                 session_uid=None):
        """
        Initialize the IQR session

        This does not initialize the working index for ranking as there are no
        known positive descriptor examples at this time.

        Adjudications
        -------------
        Adjudications are carried through between initializations. This allows
        indexed material adjudicated through-out the lifetime of the session to
        stay relevant.

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

        :param session_uid: Optional manual specification of session UUID.
        :type session_uid: str | uuid.UUID

        """
        self.uuid = session_uid or str(uuid.uuid1()).replace('-', '')
        self.lock = threading.RLock()

        self.pos_seed_neighbors = int(pos_seed_neighbors)

        # Local descriptor index for ranking, populated by a query to the
        #   nn_index instance.
        # Added external data/descriptors not added to this index.
        self.working_index = MemoryDescriptorIndex()

        # Book-keeping set so we know what positive descriptors
        # UUIDs we've used to query the neighbor index with already.
        #: :type: set[collections.Hashable]
        self._wi_seeds_used = set()

        # Descriptor elements representing data from external sources.
        #: :type: set[smqtk.representation.DescriptorElement]
        self.external_positive_descriptors = set()
        #: :type: set[smqtk.representation.DescriptorElement]
        self.external_negative_descriptors = set()

        # Descriptor references from our index (above) that have been
        #   adjudicated.
        #: :type: set[smqtk.representation.DescriptorElement]
        self.positive_descriptors = set()
        #: :type: set[smqtk.representation.DescriptorElement]
        self.negative_descriptors = set()

        # Mapping of a DescriptorElement in our relevancy search index (not the
        #   index that the nn_index uses) to the relevancy score given the
        #   recorded positive and negative adjudications.
        # This is None before any initialization or refinement occurs.
        #: :type: None | dict[smqtk.representation.DescriptorElement, float]
        self.results = None

        #
        # Algorithm Instances [+Config]
        #
        # RelevancyIndex configuration and instance that is used for producing
        #   results.
        # This is only [re]constructed when initializing the session.
        self.rel_index_config = rel_index_config
        # This is None until session initialization happens after pos/neg
        # exemplar data has been added.
        #: :type: None | smqtk.algorithms.relevancy_index.RelevancyIndex
        self.rel_index = None

    def __enter__(self):
        """
        :rtype: IqrSession
        """
        self.lock.acquire()
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()

    def ordered_results(self):
        """
        Return a tuple of the current (id, probability) result pairs in
        order of descending probability score. If there are no results yet, None
        is returned.

        :rtype: None | tuple[(smqtk.representation.DescriptorElement, float)]

        """
        with self.lock:
            if self.results:
                return tuple(sorted(six.iteritems(self.results),
                                    key=lambda p: p[1],
                                    reverse=True))
            return None

    def external_descriptors(self, positive=(), negative=()):
        """
        Add positive/negative descriptors from external data.

        These descriptors may not be a part of our working index.

        :param positive: Iterable of descriptors from external sources to
            consider positive examples.
        :type positive:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param negative: Iterable of descriptors from external sources to
            consider negative examples.
        :type negative:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        positive = set(positive)
        negative = set(negative)
        with self.lock:
            self.external_positive_descriptors.update(positive)
            self.external_positive_descriptors.difference_update(negative)

            self.external_negative_descriptors.update(negative)
            self.external_negative_descriptors.difference_update(positive)

    def adjudicate(self, new_positives=(), new_negatives=(),
                   un_positives=(), un_negatives=()):
        """
        Update current state of working index positive and negative
        adjudications based on descriptor UUIDs.

        If the same descriptor element is listed in both new positives and
        negatives, they cancel each other out, causing that descriptor to not
        be included in the adjudication.

        The given iterables must be re-traversable. Otherwise the given
        descriptors will not be properly registered.

        :param new_positives: Descriptors of elements in our working index to
            now be considered to be positively relevant.
        :type new_positives:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param new_negatives: Descriptors of elements in our working index to
            now be considered to be negatively relevant.
        :type new_negatives:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param un_positives: Descriptors of elements in our working index to now
            be considered not positive any more.
        :type un_positives:
            collections.Iterable[smqtk.representation.DescriptorElement]

        :param un_negatives: Descriptors of elements in our working index to now
            be considered not negative any more.
        :type un_negatives:
            collections.Iterable[smqtk.representation.DescriptorElement]

        """
        new_positives = set(new_positives)
        new_negatives = set(new_negatives)
        un_positives = set(un_positives)
        un_negatives = set(un_negatives)

        with self.lock:
            self.positive_descriptors.update(new_positives)
            self.positive_descriptors.difference_update(un_positives)
            self.positive_descriptors.difference_update(new_negatives)

            self.negative_descriptors.update(new_negatives)
            self.negative_descriptors.difference_update(un_negatives)
            self.negative_descriptors.difference_update(new_positives)

    def update_working_index(self, nn_index):
        """
        Initialize or update our current working index using the given
        :class:`.NearestNeighborsIndex` instance given our current positively
        labeled descriptor elements.

        We only query from the index for new positive elements since the last
        update or reset.

        :param nn_index: :class:`.NearestNeighborsIndex` to query from.
        :type nn_index: smqtk.algorithms.NearestNeighborsIndex

        :raises RuntimeError: There are no positive example descriptors in this
            session to use as a basis for querying.

        """
        pos_examples = (self.external_positive_descriptors |
                        self.positive_descriptors)
        if len(pos_examples) == 0:
            raise RuntimeError("No positive descriptors to query the neighbor "
                               "index with.")

        # Not clearing working index because this step is intended to be
        # additive.
        updated = False

        # adding to working index
        self._log.info("Building working index using %d positive examples "
                       "(%d external, %d adjudicated)",
                       len(pos_examples),
                       len(self.external_positive_descriptors),
                       len(self.positive_descriptors))
        # TODO: parallel_map and reduce with merge-dict
        for p in pos_examples:
            if p.uuid() not in self._wi_seeds_used:
                self._log.debug("Querying neighbors to: %s", p)
                self.working_index.add_many_descriptors(
                    nn_index.nn(p, n=self.pos_seed_neighbors)[0]
                )
                self._wi_seeds_used.add(p.uuid())
                updated = True

        # Make new relevancy index
        if updated:
            self._log.info("Creating new relevancy index over working index.")
            #: :type: smqtk.algorithms.relevancy_index.RelevancyIndex
            self.rel_index = plugin.from_plugin_config(
                self.rel_index_config, get_relevancy_index_impls()
            )
            self.rel_index.build_index(self.working_index.iterdescriptors())

    def refine(self):
        """ Refine current model results based on current adjudication state

        :raises RuntimeError: No working index has been initialized.
            :meth:`update_working_index` should have been called after
            adjudicating some positive examples.
        :raises RuntimeError: There are no adjudications to run on. We must
            have at least one positive adjudication.

        """
        with self.lock:
            if not self.rel_index:
                raise RuntimeError("No relevancy index yet. Must not have "
                                   "initialized session (no working index).")

            # combine pos/neg adjudications + added external data descriptors
            pos = self.positive_descriptors | self.external_positive_descriptors
            neg = self.negative_descriptors | self.external_negative_descriptors

            if not pos:
                raise RuntimeError("Did not find at least one positive "
                                   "adjudication.")

            self._log.debug("Ranking working set with %d pos and %d neg total "
                            "examples.", len(pos), len(neg))
            element_probability_map = self.rel_index.rank(pos, neg)

            if self.results is None:
                self.results = IqrResultsDict()
            self.results.update(element_probability_map)

            # Force adjudicated positives and negatives to be probability 1 and
            # 0, respectively, since we want to control where they show up in
            # our results view.
            # - Not all pos/neg descriptors may be in our working index.
            for d in pos:
                if d in self.results:
                    self.results[d] = 1.0
            for d in neg:
                if d in self.results:
                    self.results[d] = 0.0

    def reset(self):
        """ Reset the IQR Search state

        No positive adjudications, reload original feature data

        """
        with self.lock:
            self.working_index.clear()
            self._wi_seeds_used.clear()
            self.positive_descriptors.clear()
            self.negative_descriptors.clear()
            self.external_positive_descriptors.clear()
            self.external_negative_descriptors.clear()

            self.rel_index = None
            self.results = None

    ###########################################################################
    # I/O Methods

    # I/O Constants. These should not be changed.
    STATE_ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
    STATE_ZIP_FILENAME = "iqr_state.json"

    def get_state_bytes(self):
        """
        Get a byte representation of the current descriptor and adjudication
        state of this session.

        This does not encode current results or the relevancy index's state, but
        these can be reproduced with this state.

        :return: State representation bytes
        :rtype: bytes

        """
        def d_set_to_list(d_set):
            # Convert set of descriptors to list of tuples:
            #   [..., (uuid, type, vector), ...]
            return [(d.uuid(), d.type(), d.vector().tolist()) for d in d_set]

        with self:
            # Convert session descriptors into basic values.
            pos_d = d_set_to_list(self.positive_descriptors)
            neg_d = d_set_to_list(self.negative_descriptors)
            ext_pos_d = d_set_to_list(self.external_positive_descriptors)
            ext_neg_d = d_set_to_list(self.external_negative_descriptors)

        z_buffer = io.BytesIO()
        z = zipfile.ZipFile(z_buffer, 'w', self.STATE_ZIP_COMPRESSION)
        z.writestr(self.STATE_ZIP_FILENAME, json.dumps({
            'pos': pos_d,
            'neg': neg_d,
            'external_pos': ext_pos_d,
            'external_neg': ext_neg_d,
        }))
        z.close()
        return z_buffer.getvalue()

    def set_state_bytes(self, b, descriptor_factory):
        """
        Set this session's state to the given byte representation, resetting
        this session in the process.

        Bytes given must have been retrieved via a previous call to
        ``get_state_bytes`` otherwise this method will fail.

        Since this state may be completely different from the current state,
        this session is reset before applying the new state. Thus, any current
        ranking results are thrown away.

        :param b: Bytes to set this session's state to.
        :type b: bytes

        :param descriptor_factory: Descriptor element factory to use when
            generating descriptor elements from extracted data.
        :type descriptor_factory: smqtk.representation.DescriptorElementFactory

        :raises ValueError: The input bytes could not be loaded due to
            incompatibility.

        """
        z_buffer = io.BytesIO(b)
        z = zipfile.ZipFile(z_buffer, 'r', self.STATE_ZIP_COMPRESSION)
        if self.STATE_ZIP_FILENAME not in z.namelist():
            raise ValueError("Invalid bytes given, did not contain expected "
                             "zipped file name.")

        # Extract expected json file object
        state = json.loads(z.read(self.STATE_ZIP_FILENAME).decode())
        del z, z_buffer

        with self:
            self.reset()

            def load_descriptor(_uid, _type_str, vec_list):
                _e = descriptor_factory.new_descriptor(_type_str, _uid)
                if _e.has_vector():
                    assert _e.vector().tolist() == vec_list, \
                        "Found existing vector for UUID '%s' but vectors did " \
                        "not match."
                else:
                    _e.set_vector(vec_list)
                return _e

            # Read in raw descriptor data from the state, convert to descriptor
            # element, then store in our descriptor sets.
            for source, target in [(state['external_pos'],
                                    self.external_positive_descriptors),
                                   (state['external_neg'],
                                    self.external_negative_descriptors),
                                   (state['pos'], self.positive_descriptors),
                                   (state['neg'], self.negative_descriptors)]:
                for uid, type_str, vector_list in source:
                    e = load_descriptor(uid, type_str, vector_list)
                    target.add(e)
