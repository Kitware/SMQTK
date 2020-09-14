import base64
import binascii
import collections
import itertools
import json
import multiprocessing
import random
import time
import threading
import traceback
import uuid

import flask

from smqtk.algorithms import (
    Classifier,
    DescriptorGenerator,
    NearestNeighborsIndex,
    RelevancyIndex,
    SupervisedClassifier,
)
from smqtk.iqr import (
    iqr_controller,
    iqr_session,
)
from smqtk.representation import (
    ClassificationElementFactory,
    DescriptorElementFactory,
    DescriptorSet,
)
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)
from smqtk.utils.dict import merge_dict
from smqtk.web import SmqtkWebApp


def new_uuid():
    return str(uuid.uuid1(clock_seq=int(time.time() * 1000000)))\
        .replace('-', '')


def make_response_json(message, **params):
    r = {
        "message": message,
        "time": {
            "unix": time.time(),
            "utc": time.asctime(time.gmtime()),
        }
    }
    merge_dict(r, params)
    return flask.jsonify(**r)


# Get expected JSON decode exception.
#
# Flask can use one of two potential JSON parsing libraries: simplejson or
# json.  simplejson has a specific exception for decoding errors while json
# just raises a ValueError.
#
# noinspection PyProtectedMember
if hasattr(flask.json._json, 'JSONDecodeError'):
    # noinspection PyProtectedMember
    JSON_DECODE_EXCEPTION = getattr(flask.json._json, 'JSONDecodeError')
else:
    JSON_DECODE_EXCEPTION = ValueError


def parse_hashable_json_list(json_str):
    """
    Parse and check input string, looking for a JSON list of hashable values.

    :param json_str: String to parse and check.
    :type json_str: str

    :raises ValueError: Expected value check failed.

    :return: List of hashable-type values.
    :rtype: list[collections.abc.Hashable]

    """
    try:
        v_list = flask.json.loads(json_str)
    except JSON_DECODE_EXCEPTION as ex:
        raise ValueError("JSON parsing error: %s" % str(ex))
    if not isinstance(v_list, list):
        raise ValueError("JSON provided is not a list.")
    # Should not be an empty list.
    elif not v_list:
        raise ValueError("JSON list is empty.")
    # Contents of list should be numeric or string values.
    elif not all(isinstance(el, collections.abc.Hashable)
                 for el in v_list):
        raise ValueError("Not all JSON list parts were hashable values.")
    return v_list


class IqrService (SmqtkWebApp):
    """
    Configuration Notes
    -------------------
    ``descriptor_set`` will currently be configured twice: once for the
    global set and once for the nearest neighbors index. These will probably
    be the set to the same set. In more detail, the global descriptor set
    is used when the "refine" endpoint is given descriptor UUIDs
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(IqrService, cls).get_default_config()

        c_rel_index = make_default_config(
            RelevancyIndex.get_impls()
        )
        merge_dict(c_rel_index, iqr_session.DFLT_REL_INDEX_CONFIG)

        merge_dict(c, {
            "iqr_service": {

                "session_control": {
                    "positive_seed_neighbors": 500,
                    "session_expiration": {
                        "enabled": False,
                        "check_interval_seconds": 30,
                        "session_timeout": 3600,
                    }
                },

                "plugin_notes": {
                    "relevancy_index_config":
                        "The relevancy index config provided should not have "
                        "persistent storage configured as it will be used in "
                        "such a way that instances are created, built and "
                        "destroyed often.",
                    "descriptor_factory":
                        "What descriptor element factory to use when asked to "
                        "compute a descriptor on data.",
                    "descriptor_generator":
                        "Descriptor generation algorithm to use when "
                        "requested to describe data.",
                    "descriptor_set":
                        "This is the set from which given positive and "
                        "negative example descriptors are retrieved from. "
                        "Not used for nearest neighbor querying. "
                        "This set must contain all descriptors that could "
                        "possibly be used as positive/negative examples and "
                        "updated accordingly.",
                    "neighbor_index":
                        "This is the neighbor index to pull initial near-"
                        "positive descriptors from.",
                    "classifier_config":
                        "The configuration to use for training and using "
                        "classifiers for the /classifier endpoint. "
                        "When configuring a classifier for use, don't fill "
                        "out model persistence values as many classifiers "
                        "may be created and thrown away during this service's "
                        "operation.",
                    "classification_factory":
                        "Selection of the backend in which classifications "
                        "are stored. The in-memory version is recommended "
                        "because normal caching mechanisms will not account "
                        "for the variety of classifiers that can potentially "
                        "be created via this utility.",
                },

                "plugins": {
                    "relevancy_index_config": c_rel_index,
                    "descriptor_factory":
                        DescriptorElementFactory.get_default_config(),
                    "descriptor_generator": make_default_config(
                        DescriptorGenerator.get_impls()
                    ),
                    "descriptor_set": make_default_config(
                        DescriptorSet.get_impls()
                    ),
                    "neighbor_index":
                        make_default_config(NearestNeighborsIndex.get_impls()),
                    "classifier_config":
                        make_default_config(Classifier.get_impls()),
                    "classification_factory":
                        ClassificationElementFactory.get_default_config(),
                },

            }
        })
        return c

    def __init__(self, json_config):
        super(IqrService, self).__init__(json_config)
        sc_config = json_config['iqr_service']['session_control']

        # Initialize from config
        self.positive_seed_neighbors = sc_config['positive_seed_neighbors']
        self.classifier_config = \
            json_config['iqr_service']['plugins']['classifier_config']
        self.classification_factory = \
            ClassificationElementFactory.from_config(
                json_config['iqr_service']['plugins']['classification_factory']
            )

        self.descriptor_factory = DescriptorElementFactory.from_config(
            json_config['iqr_service']['plugins']['descriptor_factory']
        )

        #: :type: smqtk.algorithms.DescriptorGenerator
        self.descriptor_generator = from_config_dict(
            json_config['iqr_service']['plugins']['descriptor_generator'],
            DescriptorGenerator.get_impls(),
        )

        #: :type: smqtk.representation.DescriptorSet
        self.descriptor_set = from_config_dict(
            json_config['iqr_service']['plugins']['descriptor_set'],
            DescriptorSet.get_impls(),
        )

        #: :type: smqtk.algorithms.NearestNeighborsIndex
        self.neighbor_index = from_config_dict(
            json_config['iqr_service']['plugins']['neighbor_index'],
            NearestNeighborsIndex.get_impls(),
        )
        self.neighbor_index_lock = multiprocessing.RLock()

        self.rel_index_config = \
            json_config['iqr_service']['plugins']['relevancy_index_config']

        # Record of trained classifiers for a session. Session classifier
        # modifications locked under the parent session's global lock.
        #: :type: dict[collections.abc.Hashable, SupervisedClassifier | None]
        self.session_classifiers = {}
        # Cache of IQR session classification results on descriptors with the
        # recorded UIDs.
        # This session cache contents are purged when a classifier retrains for
        # a session.
        # Only "positive" class confidence values are retained due to the
        # binary nature of IQR-based classifiers.
        #: :type: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        self.session_classification_results = {}
        # Control for knowing when a new classifier should be trained for a
        # session (True == train new classifier). Modification for specific
        # sessions under parent session's lock.
        #: :type: dict[collections.abc.Hashable, bool]
        self.session_classifier_dirty = {}

        # Cache of random UIDs from the configured descriptor set for use
        #: :type: list[collections.abc.Hashable] | None
        self._random_uid_list_cache = None
        # Lock for mutation of this list cache
        self._random_lock = threading.RLock()

        def session_expire_callback(session):
            """
            :type session: smqtk.iqr.IqrSession
            """
            with session:
                self._log.debug("Removing session %s classifier", session.uuid)
                del self.session_classifiers[session.uuid]
                del self.session_classification_results[session.uuid]
                del self.session_classifier_dirty[session.uuid]

        self.controller = iqr_controller.IqrController(
            sc_config['session_expiration']['enabled'],
            sc_config['session_expiration']['check_interval_seconds'],
            session_expire_callback
        )
        self.session_timeout = \
            sc_config['session_expiration']['session_timeout']

        self.add_routes()

    def add_routes(self):
        """
        Setup Flask URL rules.
        """
        self.add_url_rule('/is_ready',
                          view_func=self.is_ready,
                          methods=['GET'])
        self.add_url_rule('/add_descriptor_from_data',
                          view_func=self.add_descriptor_from_data,
                          methods=['POST'])
        # TODO: Potentially other add_descriptor_from_* variants that expect
        #       other forms of input besides base64, like arbitrary URIs (to
        #       use from_uri factory function).
        self.add_url_rule('/nn_index',
                          view_func=self.get_nn_index_status,
                          methods=['GET'])
        self.add_url_rule('/nn_index',
                          view_func=self.update_nn_index,
                          methods=['POST'])
        self.add_url_rule('/nn_index',
                          view_func=self.remove_from_nn_index,
                          methods=['DELETE'])
        self.add_url_rule('/data_nearest_neighbors',
                          view_func=self.data_nearest_neighbors,
                          methods=['POST'])
        self.add_url_rule('/uid_nearest_neighbors',
                          view_func=self.uid_nearest_neighbors,
                          methods=['GET'])
        self.add_url_rule('/session_ids',
                          view_func=self.get_sessions_ids,
                          methods=['GET'])
        self.add_url_rule('/session',
                          view_func=self.get_session_info,
                          methods=['GET'])
        self.add_url_rule('/session',
                          view_func=self.init_session,
                          methods=['POST'])
        self.add_url_rule('/session',
                          view_func=self.reset_session,
                          methods=['PUT'])
        self.add_url_rule('/session',
                          view_func=self.clean_session,
                          methods=['DELETE'])
        self.add_url_rule('/add_external_pos',
                          view_func=self.add_external_positive,
                          methods=['POST'])
        self.add_url_rule('/add_external_neg',
                          view_func=self.add_external_negative,
                          methods=['POST'])
        self.add_url_rule('/adjudicate',
                          view_func=self.get_adjudication,
                          methods=['GET'])
        self.add_url_rule('/adjudicate',
                          view_func=self.adjudicate,
                          methods=['POST'])
        self.add_url_rule('/initialize',
                          view_func=self.initialize,
                          methods=['POST'])
        self.add_url_rule('/refine',
                          view_func=self.refine,
                          methods=['POST'])
        self.add_url_rule('/num_results',
                          view_func=self.num_results,
                          methods=['GET'])
        self.add_url_rule('/get_results',
                          view_func=self.get_results,
                          methods=['GET'])
        self.add_url_rule('/get_positive_adjudication_relevancy',
                          view_func=self.get_positive_adjudication_relevancy,
                          methods=['GET'])
        self.add_url_rule('/get_negative_adjudication_relevancy',
                          view_func=self.get_negative_adjudication_relevancy,
                          methods=['GET'])
        self.add_url_rule('/get_unadjudicated_relevancy',
                          view_func=self.get_unadjudicated_relevancy,
                          methods=['GET'])
        self.add_url_rule('/random_uids',
                          view_func=self.get_random_uids,
                          methods=['GET'])
        self.add_url_rule('/classify',
                          view_func=self.classify,
                          methods=['GET'])
        self.add_url_rule('/state',
                          view_func=self.get_iqr_state,
                          methods=['GET'])
        self.add_url_rule('/state',
                          view_func=self.set_iqr_state,
                          methods=['PUT'])

    def describe_base64_data(self, b64, content_type):
        """
        Compute and return the descriptor element for the given base64 data.

        The given data bytes are not retained.

        :param b64: Base64 data string.
        :type b64: str

        :param content_type: Data content type.
        :type content_type: str

        :raises TypeError: Failed to parse base64 data.

        :return: Computed descriptor element.
        :rtype: smqtk.representation.DescriptorElement
        """
        de = DataMemoryElement.from_base64(b64, content_type)
        return self.descriptor_generator.generate_one_element(
            de, descr_factory=self.descriptor_factory
        )

    # GET /is_ready
    # noinspection PyMethodMayBeStatic
    def is_ready(self):
        """
        Simple function that returns True, indicating that the server is
        active.
        """
        return make_response_json("Yes, I'm alive."), 200

    # POST /add_descriptor_from_data
    def add_descriptor_from_data(self):
        """
        Add the description of the given base64 data with content type to the
        descriptor set.

        Accept base64 data (with content type), describe it via the configured
        descriptor generator and add the resulting descriptor element to the
        configured descriptor set.

        Form Arguments:
            data_b64
                Base64-encoded input binary data to describe via
                DescriptorGenerator.  This must be of a content type accepted
                by the configured DescriptorGenerator.
            content_type
                Input data content mimetype string.

        JSON return object:
            uid
                UID of the descriptor element generated from input data
                description.  This should be equivalent to the SHA1 checksum of
                the input data.
            size
                New size (integer) of the descriptor set that has been updated
                (NOT the same as the nearest-neighbor index).

        """
        data_b64 = flask.request.form.get('data_b64', None)
        content_type = flask.request.form.get('content_type', None)
        if not data_b64:
            return make_response_json("No or empty base64 data provided."), 400
        if not content_type:
            return make_response_json("No data mimetype provided."), 400

        try:
            descriptor = self.describe_base64_data(data_b64, content_type)
        except (TypeError, binascii.Error) as e:
            if str(e) == "Incorrect padding":
                return make_response_json("Failed to parse base64 data."), 400
            # In case some other exception is raised, actually a server error.
            raise
        # Concurrent updating of descriptor set should be handled by underlying
        # implementation.
        self.descriptor_set.add_descriptor(descriptor)
        return make_response_json("Success",
                                  uid=descriptor.uuid(),
                                  size=self.descriptor_set.count()), 201

    # GET /nn_index
    def get_nn_index_status(self):
        """
        Get status/state information about the nearest-neighbor index.

        Status code 200 on success, JSON return object: {
            ...,
            // Size of the nearest-neighbor index.
            index_size=<int>
        }
        """
        with self.neighbor_index_lock:
            return (
                make_response_json("Success",
                                   index_size=self.neighbor_index.count()),
                200
            )

    # POST /nn_index
    def update_nn_index(self):
        """
        Tell the configured nearest-neighbor-index instance to update with the
        descriptors associated with the provided list of UIDs.

        This is a critical operation on the index so this method can only be
        invoked once at a time (other concurrent will block until previous
        calls have finished).

        Form Arguments:
            descriptor_uids
                JSON list of UID strings.  If one or more UIDs do not match
                descriptors in our current descriptor-set we return an error
                message.

        JSON return object:
            message
                Success string
            descriptor_uids
                List of UIDs the neighbor index was updated with.  This should
                be congruent with the list provided.
            index_size
                New size of the nearest-neighbors index.

        """
        descr_uid_str = flask.request.form.get('descriptor_uids', None)
        if not descr_uid_str:  # empty string or None
            return make_response_json("No descriptor UID JSON provided."), 400

        # Load and check JSON input.
        try:
            descr_uid_list = parse_hashable_json_list(descr_uid_str)
        except ValueError as ex:
            return make_response_json("%s" % str(ex)), 400

        with self.neighbor_index_lock:
            try:
                # KeyError may not occur until returned iterator is iterated.
                descr_elems = \
                    self.descriptor_set.get_many_descriptors(descr_uid_list)
                self.neighbor_index.update_index(descr_elems)
            except KeyError:
                # Some UIDs are not present in the current index.  Isolate
                # which UIDs are not contained.
                uids_not_ingested = []
                for uid in descr_uid_list:
                    if not self.descriptor_set.has_descriptor(uid):
                        uids_not_ingested.append(uid)
                return make_response_json("Some provided UIDs do not exist in "
                                          "the current index.",
                                          bad_uids=uids_not_ingested), 400

            return (
                make_response_json("Success",
                                   descriptor_uids=descr_uid_list,
                                   index_size=self.neighbor_index.count()),
                200
            )

    # DELETE /nn_index
    def remove_from_nn_index(self):
        """
        Remove descriptors from the nearest-neighbors index given their UIDs.

        Receive one or more descriptor UIDs, that exist in the NN-index, that
        are to be removed from the NN-index.  This DOES NOT remove elements
        from the global descriptor set.

        This is a critical operation on the index so this method can only be
        invoked once at a time (other concurrent will block until previous
        calls have finished).

        Form Arguments:
            descriptor_uids
                JSON list of descriptor UIDs to remove from the nearest-
                neighbor index.  These UIDs must be present in the index,
                otherwise an 404 error is returned.

        Status code 200 on success, JSON return object: {
            ...,

            // List of UID values removed from the index.
            descriptor_uids=<list[str]>,

            // New size of the nearest-neighbors index.
            index_size=<int>
        }

        """
        descr_uid_str = flask.request.values.get('descriptor_uids', None)
        if not descr_uid_str:  # empty string or None
            return make_response_json("No descriptor UID JSON provided."), 400

        # Load and check JSON input
        try:
            descr_uid_list = parse_hashable_json_list(descr_uid_str)
        except ValueError as ex:
            return make_response_json("%s" % str(ex)), 400

        with self.neighbor_index_lock:
            try:
                # empty list already checked for in above try-catch, so we
                # should never see a ValueError here.  KeyError still possible.
                self.neighbor_index.remove_from_index(descr_uid_list)
            except KeyError as ex:
                return make_response_json("Some provided UIDs do not exist in "
                                          "the current index.",
                                          bad_uids=ex.args), 400

            return (
                make_response_json("Success",
                                   descriptor_uids=descr_uid_list,
                                   index_size=self.neighbor_index.count()),
                200
            )

    # POST /data_nearest_neighbors
    def data_nearest_neighbors(self):
        """
        Take in data in base64 encoding with a mimetype and find its 'k'
        nearest neighbors according to the current index, including their
        distance values (metric determined by nearest-neighbors-index algorithm
        configuration).

        This endpoint does not need a session ID due to the
        nearest-neighbor-index being a shared resource across IQR sessions.

        Form Arguments:
            data_b64
                Base64-encoded input binary data to describe via
                DescriptorGenerator.  This must be of a content type accepted
                by the configured DescriptorGenerator.
            content_type
                Input data content mimetype string.
            k
                Integer number of nearest neighbor descriptor UIDs to return
                along with their distances.

        JSON return object:
            neighbor_uids
                Ordered list of neighbor UID values. Index 0 represents the
                closest neighbor while the last index represents the farthest
                neighbor.  Parallel in relationship to `neighbor_dists`.
            neighbor_dists
                Ordered list of neighbor distance values. Index 0 represents
                the closest neighbor while the last index represents the
                farthest neighbor.  Parallel in relationship to
                'neighbor_uids`.

        """
        data_b64 = flask.request.form.get('data_b64', None)
        content_type = flask.request.form.get('content_type', None)
        k_str = flask.request.form.get('k', None)
        if not data_b64:
            return make_response_json("No or empty base64 data provided."), 400
        if not content_type:
            return make_response_json("No data mimetype provided."), 400
        if not k_str:
            return make_response_json("No 'k' value provided."), 400

        try:
            k = int(k_str)
        except ValueError as ex:
            return make_response_json("Failed to convert 'k' argument to an "
                                      "integer: %s" % str(ex)), 400

        try:
            descriptor = self.describe_base64_data(data_b64, content_type)
        except (TypeError, binascii.Error) as e:
            return make_response_json("Failed to parse base64 data: {}"
                                      .format(str(e))), 400

        n_elems, n_dists = self.neighbor_index.nn(descriptor, k)
        return make_response_json("Success",
                                  neighbor_uids=[e.uuid() for e in n_elems],
                                  neighbor_dists=[d for d in n_dists]), 200

    # GET /uid_nearest_neighbors
    def uid_nearest_neighbors(self):
        """
        Take in the UID that matches an ingested descriptor and find that
        descriptor's 'k' nearest neighbors according to the current index,
        including their distance values (metric determined by
        nearest-neighbors-index algorithm configuration).

        This endpoint does not need a session ID due to the
        nearest-neighbor-index being a shared resource across IQR sessions.

        This endpoint can be more advantageous compared the
        `data_nearest_neighbors` endpoint if you know a descriptor has already
        been ingested (via `add_descriptor_from_data` or otherwise) as a
        potentially new descriptor does not have to be computed.

        URL Arguments:
            uid
                UID of the descriptor to get the nearest neighbors for.  This
                should also match the SHA1 checksum of the data being
                described.
            k
                Integer number of nearest neighbor descriptor UIDs to return
                along with their distances.

        JSON return object:
            neighbor_uids
                Ordered list of neighbor UID values. Index 0 represents the
                closest neighbor while the last index represents the farthest
                neighbor.  Parallel in relationship to `neighbor_dists`.
            neighbor_dists
                Ordered list of neighbor distance values. Index 0 represents
                the closest neighbor while the last index represents the
                farthest neighbor.  Parallel in relationship to
                'neighbor_uids`.

        """
        uid = flask.request.values.get('uid', None)
        k_str = flask.request.values.get('k', None)
        if not uid:
            return make_response_json("No UID provided."), 400
        if not k_str:
            return make_response_json("No 'k' value provided."), 400

        try:
            k = int(k_str)
        except ValueError as ex:
            return make_response_json("Failed to convert 'k' argument to an "
                                      "integer: %s" % str(ex)), 400

        try:
            descriptor = self.descriptor_set.get_descriptor(uid)
        except KeyError:
            return make_response_json("Failed to get descriptor for UID %s"
                                      % uid), 400

        n_elems, n_dists = self.neighbor_index.nn(descriptor, k)
        return make_response_json("Success",
                                  neighbor_uids=[e.uuid() for e in n_elems],
                                  neighbor_dists=[d for d in n_dists]), 200

    # GET /session_ids
    def get_sessions_ids(self):
        """
        Get the list of current, active session IDs.
        """
        session_uuids = self.controller.session_uuids()
        return make_response_json("Current session UUID values",
                                  session_uuids=session_uuids), 200

    # GET /session
    def get_session_info(self):
        """
        Get a JSON return with session state information.

        URL Arguments:
            sid
                ID of the session.

        Return JSON:
            sid (str):
                Input IQR Session UID.
            uuids_pos (list[str]):
                List of working-set descriptor UIDs that are currently
                adjudicated positive.
            uuids_neg (list[str]):
                List of working-set descriptor UIDs that are currently
                adjudicated negative.
            uuids_pos_ext (list[str]):
                List of descriptor UIDs from external data that are currently
                adjudicated positive.
            uuids_neg_ext (list[str]):
                List of descriptor UIDs from external data that are currently
                adjudicated negative.
            uuids_pos_in_model (list[str]):
                List of working-set descriptor UIDs that adjudicated positive
                at the time of this session's last refinement.
            uuids_neg_in_model (list[str]):
                List of working-set descriptor UIDs that adjudicated negative
                at the time of this session's last refinement.
            uuids_pos_ext_in_model (list[str]):
                List of descriptor UIDs from external data that were
                adjudicated positive at the time of this session's last
                refinement.
            uuids_neg_ext_in_model (list[str]):
                List of descriptor UIDs from external data that were
                adjudicated negative at the time of this session's last
                refinement.
            wi_count (int):
                Number of elements currently in the working index to be ranked
                by session refinement.
        """
        sid = flask.request.args.get('sid', None)
        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            uuids_pos = [d.uuid() for d in iqrs.positive_descriptors]
            uuids_pos_external = [d.uuid() for d in
                                  iqrs.external_positive_descriptors]
            uuids_neg = [d.uuid() for d in iqrs.negative_descriptors]
            uuids_neg_external = [d.uuid() for d in
                                  iqrs.external_negative_descriptors]
            uids_pos_in_model = [d.uuid() for d in iqrs.rank_contrib_pos]
            uids_pos_ext_in_model = [d.uuid() for d
                                     in iqrs.rank_contrib_pos_ext]
            uids_neg_in_model = [d.uuid() for d in iqrs.rank_contrib_neg]
            uids_neg_ext_in_model = [d.uuid() for d
                                     in iqrs.rank_contrib_neg_ext]
            wi_count = iqrs.working_set.count()
        finally:
            iqrs.lock.release()

        return make_response_json("Session '%s' info" % sid,
                                  sid=sid,
                                  uuids_pos=uuids_pos,
                                  uuids_neg=uuids_neg,
                                  uuids_pos_ext=uuids_pos_external,
                                  uuids_neg_ext=uuids_neg_external,
                                  uuids_pos_in_model=uids_pos_in_model,
                                  uuids_pos_ext_in_model=uids_pos_ext_in_model,
                                  uuids_neg_in_model=uids_neg_in_model,
                                  uuids_neg_ext_in_model=uids_neg_ext_in_model,
                                  wi_count=wi_count), 200

    # POST /session
    def init_session(self):
        """
        Initialize a new session in the controller.

        Form args:
            sid
                Optional UUID string specification of session to initialize. If
                one is not provided, we create one here.

        """
        sid = flask.request.form.get('sid', None)
        if sid is None:
            sid = new_uuid()

        if self.controller.has_session_uuid(sid):
            return make_response_json(
                "Session with id '%s' already exists" % sid,
                sid=sid,
            ), 409  # CONFLICT

        iqrs = iqr_session.IqrSession(self.positive_seed_neighbors,
                                      self.rel_index_config,
                                      sid)
        with self.controller:
            with iqrs:  # because classifier maps locked by session
                self.controller.add_session(iqrs, self.session_timeout)
                self.session_classifiers[sid] = None
                self.session_classification_results[sid] = {}
                self.session_classifier_dirty[sid] = True

        return make_response_json("Created new session with ID '%s'" % sid,
                                  sid=sid), 201  # CREATED

    # PUT /session
    def reset_session(self):
        """
        Reset an existing session. This does not remove the session, so actions
        may

        Form args:
            sid
                UUID (string) of the session to reset.

        """
        sid = flask.request.form.get('sid', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            iqrs.reset()
            self.session_classifiers[sid] = None
            self.session_classification_results[sid] = {}
            self.session_classifier_dirty[sid] = True

        finally:
            iqrs.lock.release()

        return make_response_json("Reset IQR session '%s'" % sid,
                                  sid=sid), 200

    # DELETE /session
    def clean_session(self):
        """
        Clean resources associated with the session of the given UUID.

        Similar to session resetting, but this also removed the session
        resource causing future references without another initialize to return
        errors.

        Form args:
            sid
                UUID of the session to clean.

        """
        sid = flask.request.form.get('sid', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            with self.controller.get_session(sid) as iqrs:
                iqrs.reset()
                del self.session_classifiers[sid]
                del self.session_classification_results[sid]
                del self.session_classifier_dirty[sid]
            self.controller.remove_session(sid)
        return make_response_json("Cleaned session resources for '%s'" % sid,
                                  sid=sid), 200

    # POST /add_external_pos
    def add_external_positive(self):
        """
        Describe the given data and store as a positive example from external
        data.

        Form args:
            sid
                The id of the session to add the generated descriptor to.
            base64
                The base64 byes of the data. This should use the standard or
                URL-safe alphabet as the python ``base64.urlsafe_b64decode``
                module function would expect (handles either alphabet).
            content_type
                The mimetype of the bytes given.

        A return JSON along with a code 201 means a descriptor was successfully
        computed and added to the session external positives. The returned JSON
        includes a reference to the UUID of the descriptor computed under the
        ``descr_uuid`` key.

        """
        sid = flask.request.form.get('sid', None)
        data_base64 = flask.request.form.get('base64', None)
        data_content_type = flask.request.form.get('content_type', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400
        if not data_base64:
            return make_response_json("No or empty base64 data provided."), 400
        if not data_content_type:
            return make_response_json("No data mimetype provided."), 400

        descriptor = self.describe_base64_data(data_base64, data_content_type)

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            iqrs.external_descriptors(positive=[descriptor])
            self._log.debug("[%s] session Classifier dirty", sid)
            self.session_classifier_dirty[sid] = True
        finally:
            iqrs.lock.release()

        return make_response_json("Success", descr_uuid=descriptor.uuid()), 201

    # POST /add_external_neg
    def add_external_negative(self):
        """
        Describe the given data and store as a negative example from external
        data.

        Form args:
            sid
                The id of the session to add the generated descriptor to.
            base64
                The base64 byes of the data. This should use the standard or
                URL-safe alphabet as the python ``base64.urlsafe_b64decode``
                module function would expect (handles either alphabet).
            content_type
                The mimetype of the bytes given.

        A return JSON along with a code 201 means a descriptor was successfully
        computed and added to the session external negatives. The returned JSON
        includes a reference to the UUID of the descriptor computed under the
        ``descr_uuid`` key.

        """
        sid = flask.request.form.get('sid', None)
        data_base64 = flask.request.form.get('base64', None)
        data_content_type = flask.request.form.get('content_type', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400
        if not data_base64:
            return make_response_json("No or empty base64 data provided."), 400
        if not data_content_type:
            return make_response_json("No data mimetype provided."), 400

        descriptor = self.describe_base64_data(data_base64,
                                               data_content_type)

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            iqrs.external_descriptors(negative=[descriptor])
            self._log.debug("[%s] session Classifier dirty", sid)
            self.session_classifier_dirty[sid] = True
        finally:
            iqrs.lock.release()

        return make_response_json("Success",
                                  descr_uuid=descriptor.uuid()), 201

    # GET /adjudicate
    def get_adjudication(self):
        """
        Get the adjudication state of a descriptor given its UID.

        Form args:
            sid
                Session Id.
            uid
                Query descriptor UID.
        """
        sid = flask.request.args.get('sid', None)
        uid = flask.request.args.get('uid', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400
        elif uid is None:
            return make_response_json("No descriptor uid provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            all_pos = (iqrs.external_positive_descriptors |
                       iqrs.positive_descriptors)
            all_neg = (iqrs.external_negative_descriptors |
                       iqrs.negative_descriptors)

        finally:
            iqrs.lock.release()

        is_pos = uid in {d.uuid() for d in all_pos}
        is_neg = uid in {d.uuid() for d in all_neg}

        if is_pos and is_neg:
            return make_response_json("UID slotted as both positive and "
                                      "negative?"), 500

        return make_response_json("%s descriptor adjudication" % uid,
                                  is_pos=is_pos, is_neg=is_neg), 200

    # POST /adjudicate
    def adjudicate(self):
        """
        Incrementally update internal adjudication state given new positives
        and negatives, and optionally IDs for descriptors now marked neutral.

        If the same UUID is present in both positive and negative sets, they
        cancel each other out (remains neutral).

        Descriptor uuids that may be provided must be available in the
        configured descriptor set.

        Form Args:
            sid
                UUID of session to interact with.
            pos
                List of descriptor UUIDs that should be marked positive.
            neg
                List of descriptor UUIDs that should be marked negative.
            neutral
                List of descriptor UUIDs that should not be marked positive or
                negative. If a UUID present in this list and in either pos or
                neg will be considered neutral.

        """
        sid = flask.request.form.get('sid', None)
        pos_uuids = flask.request.form.get('pos', '[]')
        neg_uuids = flask.request.form.get('neg', '[]')
        neu_uuids = flask.request.form.get('neutral', '[]')

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        pos_uuids = set(json.loads(pos_uuids))
        neg_uuids = set(json.loads(neg_uuids))
        neu_uuids = set(json.loads(neu_uuids))

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            self._log.debug("Determining UIDs for query.")
            # Reduce number of UIDs we query the descriptor set for based on
            # what we already have in our pos/neg adjudication sets.
            have_elems = iqrs.positive_descriptors | iqrs.negative_descriptors
            have_elem_map = {e.uuid(): e for e in have_elems}
            have_uuids = set(have_elem_map.keys())
            # UIDs for elements we don't have yet
            q_pos_uids = pos_uuids.difference(have_uuids)
            q_neg_uids = neg_uuids.difference(have_uuids)
            q_neu_uids = neu_uuids.difference(have_uuids)
            # Elements for UIDs we do already have
            have_pos_elems = {have_elem_map[uid] for uid
                              in pos_uuids.intersection(have_uuids)}
            have_neg_elems = {have_elem_map[uid] for uid
                              in neg_uuids.intersection(have_uuids)}
            have_neu_elems = {have_elem_map[uid] for uid
                              in neu_uuids.intersection(have_uuids)}

            # Combine UID sets into one in order to make a single
            # descriptor-set query
            # - get_many_descriptors can raise KeyError
            self._log.debug("Getting the descriptors for UUIDs")
            descr_iter = self.descriptor_set.get_many_descriptors(
                itertools.chain(q_pos_uids, q_neg_uids, q_neu_uids)
            )
            self._log.debug("- Slicing out positive descriptors...")
            pos_d = set(itertools.chain(
                have_pos_elems,
                itertools.islice(descr_iter, len(q_pos_uids)))
            )
            assert len(pos_d) == len(pos_uuids), \
                "Input pos UIDs doesn't match result descriptors set: " \
                "{} != {}".format(len(pos_d), len(pos_uuids))
            assert set(d.uuid() for d in pos_d) == pos_uuids, \
                "Result positive descriptor element UIDs don't match input " \
                "uids."

            self._log.debug("- Slicing out negative descriptors...")
            neg_d = set(itertools.chain(
                have_neg_elems,
                itertools.islice(descr_iter, len(q_neg_uids))
            ))
            assert len(neg_d) == len(neg_uuids), \
                "Input neg UIDs doesn't match result descriptors set: " \
                "{} != {}".format(len(neg_d), len(neg_uuids))
            assert set(d.uuid() for d in neg_d) == neg_uuids, \
                "Result negative descriptor element UIDs don't match input " \
                "uids."

            self._log.debug("- Slicing out neutral descriptors...")
            neu_d = set(itertools.chain(
                have_neu_elems,
                itertools.islice(descr_iter, len(q_neu_uids))
            ))
            assert len(neu_d) == len(neu_uuids), \
                "Input neu UIDs doesn't match result descriptors set: " \
                "{} != {}".format(len(neu_d), len(neu_uuids))
            assert set(d.uuid() for d in neu_d) == neu_uuids, \
                "Result neutral descriptor element UIDs don't match input " \
                "uids."
            self._log.debug("Getting the descriptors for UUIDs -- Done")

            # Record previous pos/neg descriptors via shallow copy for
            # determining if an existing classifier is dirty after this
            # adjudication.
            orig_pos = set(iqrs.positive_descriptors)
            orig_neg = set(iqrs.negative_descriptors)

            self._log.debug("[%s] Adjudicating", sid)
            iqrs.adjudicate(pos_d, neg_d, neu_d, neu_d)

            # Flag classifier as dirty if change in pos/neg sets
            diff_pos = \
                iqrs.positive_descriptors.symmetric_difference(orig_pos)
            diff_neg = \
                iqrs.negative_descriptors.symmetric_difference(orig_neg)
            if diff_pos or diff_neg:
                self._log.debug("[%s] session Classifier dirty", sid)
                self.session_classifier_dirty[sid] = True

        except KeyError as ex:
            err_uuid = str(ex)
            self._log.warning(traceback.format_exc())
            return make_response_json(
                "Descriptor UUID '%s' cannot be found in the "
                "configured descriptor set."
                % err_uuid,
                sid=sid,
                uuid=err_uuid,
            ), 404

        finally:
            iqrs.lock.release()

        return make_response_json(
            "Finished adjudication",
            sid=sid,
        ), 200

    # POST /initialize
    def initialize(self):
        """
        Update the working set based on the currently positive examples and
        adjudications.

        Form Arguments:
            sid
                Id of the session to update.
        """
        sid = flask.request.form.get('sid', None)
        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid,
                                          success=False), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            iqrs.update_working_set(self.neighbor_index)
        except RuntimeError as ex:
            if "No positive descriptors to query" in str(ex):
                return make_response_json("Failed to initialize, no positive "
                                          "descriptors to query",
                                          sid=sid, success=False), 200
            else:
                raise
        finally:
            iqrs.lock.release()

        return make_response_json("Success", sid=sid, success=True), 200

    # POST /refine
    def refine(self):
        """
        (Re)Create ranking of working set content by order of relevance to
        examples and adjudications.

        Form args:
            sid
                Id of the session to use.

        """
        sid = flask.request.form.get('sid', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id %s not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            self._log.info("[%s] Refining", sid)
            iqrs.refine()
        except RuntimeError as ex:
            ex_s = str(ex)
            if "No relevancy index yet." in ex_s:
                return make_response_json("No initialization has occurred yet "
                                          "for this IQR session."), 400
            raise
        finally:
            iqrs.lock.release()

        return make_response_json("Refine complete", sid=sid), 201

    # GET /num_results
    def num_results(self):
        """
        Get the total number of results that have been ranked.

        This is usually 0 before refinement and the size of the working set
        after refinement.

        URI Args:
            sid
                UUID of the session to use

        """
        sid = flask.request.args.get('sid', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            if iqrs.results:
                size = len(iqrs.results)
            else:
                size = 0

        finally:
            iqrs.lock.release()

        return make_response_json("Currently %d results for session %s"
                                  % (size, sid),
                                  num_results=size,
                                  sid=sid), 200

    # GET /get_results
    def get_results(self):
        """
        Get the relevancy score for working set descriptor elements between
        the optionally specified offset and limit indices, ordered by
        *descending* predicted relevancy values (in [0, 1] range).

        If ``i`` (offset, inclusive) is omitted, we assume a starting index of
        0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
        is the same as the number of results available.

        If the requested session has not been refined yet (no ranking), an
        empty results list is returned.

        URL Args:
            sid: str
                UUID of the session to use
            i: int
                Starting index (inclusive)
            j: int
                Ending index (exclusive)

        Possible error code returns:
            400
                No session ID provided. Offset/limit index values were not
                valid integers.
            404
                No session for the given ID.

        Success returns 201 and a JSON object that includes the keys:
            sid: str
                String IQR session ID accessed.
            i: int
                Index offset used.
            j: int
                Index limit used.
            total_results: int
                Total number of ranked results with predicted relevancy. This
                is not necessarily the number of results returned from the
                call due to the optional use of ``i``  and ``j``.
            results: list[(str, float)]
                A list of ``(element_id, probability)`` pairs. The
                ``element_id`` is the UUID of the data/descriptor the result
                relevancy probability score is associated do. The
                ``probability`` value is a float in the [0, 1] range.

        """
        sid = flask.request.args.get('sid', None)
        i = flask.request.args.get('i', None)
        j = flask.request.args.get('j', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            ordered_results = iqrs.ordered_results()
            num_results = len(ordered_results)
            # int() can raise ValueError, catch
            i = 0 if i is None else int(i)
            j = num_results if j is None else int(j)
            # We ensured i, j are valid by this point
            r = [[d.uuid(), prob] for d, prob in ordered_results[i:j]]
        except ValueError:
            return make_response_json("Invalid bounds index value(s)"), 400

        finally:
            iqrs.lock.release()

        return make_response_json("Returning result pairs",
                                  sid=sid, i=i, j=j,
                                  total_results=num_results,
                                  results=r), 200

    # GET /get_positive_adjudication_relevancy
    def get_positive_adjudication_relevancy(self):
        """
        Get the relevancy scores for positively adjudicated elements in the
        working set between the optionally provided index offset and limit,
        ordered by *descending* predicted relevancy values (in [0, 1] range).

        If ``i`` (offset, inclusive) is omitted, we assume a starting index of
        0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
        is the same as the number of results available.

        If the requested session has not been refined yet (no ranking), an
        empty results list is returned.

        URI Args:
            sid: str
                UUID of the IQR session to use.
            i: int
                Starting index (inclusive).
            j: int
                Ending index (exclusive).

        Possible error code returns:
            400
                No session ID provided. Offset/limit index values were not
                valid integers.
            404
                No session for the given ID.

        Returns 200 and a JSON object that includes the following:
            sid: str
                String IQR session ID accessed.
            i: str
                Index offset used.
            j: str
                Index limit used.
            total: int
                Total number of positive adjudications in the current IQR
                session.
            results: list[(str, float)]
                List of ``(uuid, score)`` tuples for positively adjudicated
                descriptors in the working index, ordered by descending score.

        """
        sid = flask.request.args.get('sid', None)
        i = flask.request.args.get('i', None)
        j = flask.request.args.get('j', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            pos_results = iqrs.get_positive_adjudication_relevancy()
            num_pos = len(pos_results)
            # int() can raise ValueError, catch
            i = 0 if i is None else int(i)
            j = num_pos if j is None else int(j)
            r = [[d.uuid(), prob] for d, prob in pos_results[i:j]]
        except ValueError:
            return make_response_json("Invalid bounds index value(s)"), 400
        finally:
            iqrs.lock.release()

        return make_response_json(
            "success", sid=sid, i=i, j=j,
            total=num_pos, results=r
        ), 200

    # GET /get_negative_adjudication_relevancy
    def get_negative_adjudication_relevancy(self):
        """
        Get the relevancy scores for negatively adjudicated elements in the
        working set between the optionally provided offset and limit,
        ordered by *descending* predicted relevancy values (in [0, 1] range).

        If ``i`` (offset, inclusive) is omitted, we assume a starting index of
        0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
        is the same as the number of results available.

        If the requested session has not been refined yet (no ranking), an
        empty results list is returned.

        URI Args:
            sid: str
                UUID of the IQR session to use.
            i: int
                Starting index (inclusive).
            j: int
                Ending index (exclusive).

        Possible error code returns:
            400
                No session ID provided. Offset/limit index values were not
                valid integers.
            404
                No session for the given ID.

        Returns 200 and a JSON object that includes the following:
            sid: str
                String IQR session ID accessed.
            i: int
                Index offset used.
            j: int
                Index limit used.
            total: int
                Total number of negative adjudications in the current IQR
                session.
            results: list[(str, float)]
                List of ``(uuid, score)`` tuples for negatively adjudicated
                descriptors in the working set, ordered by descending score.

        """
        sid = flask.request.args.get('sid', None)
        i = flask.request.args.get('i', None)
        j = flask.request.args.get('j', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            neg_results = iqrs.get_negative_adjudication_relevancy()
            num_neg = len(neg_results)
            # int() can raise ValueError, catch
            i = 0 if i is None else int(i)
            j = num_neg if j is None else int(j)
            r = [[d.uuid(), prob] for d, prob in neg_results[i:j]]
        except ValueError:
            return make_response_json("Invalid bounds index value(s)"), 400
        finally:
            iqrs.lock.release()

        return make_response_json(
            "success", sid=sid, i=i, j=j,
            total=num_neg, results=r
        ), 200

    # GET /get_unadjudicated_relevancy
    def get_unadjudicated_relevancy(self):
        """
        Get the relevancy scores for non-adjudicated elements in the working
        set between the optionally provided index offset and limit, ordered
        by descending predicted relevancy value ([0, 1] range).

        If ``i`` (offset, inclusive) is omitted, we assume a starting index of
        0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
        is the same as the number of results available.

        If the requested session has not been refined yet (no ranking), an
        empty results list is returned.

        URI Args:
            sid: str
                UUID of the IQR session to use.
            i: int
                Starting index (inclusive).
            j: int
                Ending index (exclusive).

        Possible error code returns:
            400
                No session ID provided. Offset/limit index values were not
                valid integers.
            404
                No session for the given ID.

        Returns 200 and a JSON object that includes the following:
            sid: str
                String IQR session ID accessed.
            i: int
                Index offset used.
            j: int
                Index limit used.
            total: int
                Total number of negative adjudications in the current IQR
                session.
            results: list[(str, float)]
                List of ``(uuid, score)`` tuples for negatively adjudicated
                descriptors in the working set, ordered by descending score.

        """
        sid = flask.request.args.get('sid', None)
        i = flask.request.args.get('i', None)
        j = flask.request.args.get('j', None)
        # TODO: Add optional parameter that is used instead of i/j that causes
        #       the return of N uniformly distributed examples in this result
        #       subset based on relevancy score.
        #       - Add on IqrSession class side?

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            unadj_ordered = iqrs.get_unadjudicated_relevancy()
            total = len(unadj_ordered)
            # int() can raise ValueError, catch
            i = 0 if i is None else int(i)
            j = total if j is None else int(j)
            r = [[d.uuid(), prob] for d, prob in unadj_ordered[i:j]]
        except ValueError:
            return make_response_json("Invalid bounds index value(s)"), 400
        finally:
            iqrs.lock.release()

        return make_response_json(
            "success", sid=sid, i=i, j=j,
            total=total, results=r
        ), 200

    def get_random_uids(self):
        """
        Get a slice of random descriptor UIDs from the global set between the
        optionally provided index offset and limit.

        If ``i`` (offset, inclusive) is omitted, we assume a starting index of
        0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
        is the same as the number of results available.

        URI Args:
            i: int
                Starting index (inclusive). 0 by default.
            j: int
                Ending index (exclusive). Total global index size by default.
            refresh: bool
                If `true` we refresh our random UID list from the global index.
                Otherwise this when `false` we utilize the same globally cached
                random ordering for pagination stability.

        Returns 200 and a JSON object that includes the following:
            results: list[str]
                List of string descriptor UIDs from the global set within the
                give `[i, j]` slice.
            total: int
                Total number of UIDs in the global set.
        """
        i = flask.request.args.get('i', 0)
        j = flask.request.args.get('j', None)
        refresh_str = flask.request.args.get('refresh', 'false')

        try:
            refresh = json.loads(refresh_str)
        except json.JSONDecodeError:
            return make_response_json("Value for 'refresh' should be a valid "
                                      "JSON boolean."), 400

        with self._random_lock:
            if self._random_uid_list_cache is None or refresh:
                self._random_uid_list_cache = list(self.descriptor_set.keys())
                random.shuffle(self._random_uid_list_cache)
            total = len(self._random_uid_list_cache)
            try:
                i = int(i)
                j = total if j is None else int(j)
                results = self._random_uid_list_cache[i:j]
            except ValueError:
                return make_response_json("Invalid bounds index value(s)"), 400

        return make_response_json(
            "success", total=total, results=results
        ), 200

    def _ensure_session_classifier(self, iqrs)  :
        """
        Return the binary pos/neg classifier for this session.

        If no classifier exists yet for this session, or it has been marked
        dirty, retrain the classifier based on the input classifier
        configuration.

        This method assumes its being executed within an IQR session lock.

        :param smqtk.iqr.IqrSession iqrs:
            UUID of the IQR session to use.

        :return:
            Binary classifier for the given IQR session, the positive
            classification label and the negative classification label.
        :rtype: (smqtk.algorithms.SupervisedClassifier, str, str)
        """
        sid = iqrs.uuid

        all_pos = (iqrs.external_positive_descriptors |
                   iqrs.positive_descriptors)
        if not all_pos:
            raise RuntimeError("No positive labels in current IQR session. "
                               "Required for a supervised classifier.")
        all_neg = (iqrs.external_negative_descriptors |
                   iqrs.negative_descriptors)
        if not all_neg:
            raise RuntimeError("No negative labels in current IQR session. "
                               "Required for a supervised classifier.")

        classifier = self.session_classifiers.get(sid, None)

        pos_label = "positive"
        neg_label = "negative"

        if self.session_classifier_dirty[sid] or classifier is None:
            self._log.debug("Training new classifier for current "
                            "adjudication state...")

            #: :type: SupervisedClassifier
            classifier = from_config_dict(
                self.classifier_config,
                SupervisedClassifier.get_impls()
            )
            classifier.train(
                {pos_label: all_pos,
                 neg_label: all_neg}
            )

            self.session_classifiers[sid] = classifier
            self.session_classification_results[sid] = {}
            self.session_classifier_dirty[sid] = False

        return classifier, pos_label, neg_label

    # GET /classify
    def classify(self):
        """
        Given a refined session ID and some number of descriptor UUIDs, create
        a classifier according to the current state and classify the given
        descriptors adjudicated.

        This will fail if the session has not been given adjudications
        (refined) yet.

        URI Args:
            sid
                UUID of the session to utilize
            uuids
                List of descriptor UUIDs to classify. These UUIDs must
                associate to descriptors in the configured descriptor set.

        TODO: Optionally take in a list of JSON objects encoding base64 bytes
              and content type of raw data to describe and then classify, thus
              extending classification ability to arbitrary new data.

        """
        # Record clean/dirty status after making classifier/refining so we
        # don't train a new classifier when we don't have to.
        sid = flask.request.args.get('sid', None)
        uuids = flask.request.args.get('uuids', None)

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        try:
            uuids = json.loads(uuids)
        except ValueError:
            return make_response_json(
                "Failed to decode uuids as json. Given '%s'"
                % uuids
            ), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                self._log.warning("No IQR Session with UID '{}' found."
                                  .format(sid))
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            try:
                classifier, pos_label, neg_label = \
                    self._ensure_session_classifier(iqrs)
            except RuntimeError as ex:
                # Classification training may have failed.
                return make_response_json(
                    str(ex), sid=sid
                ), 400

            # Reduce descriptors actively classified to those not represented
            # in the results cache.
            c_cache = self.session_classification_results[sid]
            uuid_for_clsify = set(uuids) - set(c_cache)
            if uuid_for_clsify:
                # Get descriptor elements for classification
                # get_many_descriptors can raise KeyError
                descriptors = list(self.descriptor_set
                                   .get_many_descriptors(uuid_for_clsify))
                classifications = classifier.classify_elements(
                    descriptors, self.classification_factory,
                    # TODO: overwrite? ensure memory only elements?
                )

                # Update cache
                for c in classifications:
                    c_cache[c.uuid] = c[pos_label]
            elif uuids:
                self._log.info("No classifications necessary, using cache.")

            # Format output to be parallel lists of UUIDs input and
            # positive class classification scores.
            o_uuids = uuids
            o_proba = [c_cache[uid] for uid in uuids]

        except KeyError as ex:
            err_uuid = str(ex)
            self._log.warning(traceback.format_exc())
            return make_response_json(
                "Descriptor UUID '%s' cannot be found in the "
                "configured descriptor set."
                % err_uuid,
                sid=sid,
                uuid=err_uuid,
            ), 404

        finally:
            iqrs.lock.release()

        return make_response_json(
            "Finished classification",
            sid=sid,
            uuids=o_uuids,
            proba=o_proba,
        ), 200

    # TODO: Save/Export classifier model/state/configuration?

    # GET /state
    def get_iqr_state(self):
        """ [See api.rst]
        Create and return a binary package representing this IQR session's
        state.

        An IQR state is composed of the descriptor vectors, and their UUIDs,
        that were added from external sources, or were adjudicated, positive
        and negative.

        URL Arguments:
            sid
                Session ID to get the state of.

        Possible error code returns:
            400
                No session ID provided.
            404
                No session for the given ID.

        Success returns 200: {
            message = "Success"
            ...
            sid = <str>
            state_b64 = <str>
        }

        """
        sid = flask.request.args.get('sid', None)

        if sid is None:
            return make_response_json("No session id (sid) provided."), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            iqrs_state_bytes = iqrs.get_state_bytes()
        finally:
            iqrs.lock.release()

        # Convert state bytes to a base64 string
        # - `b64encode` returns bytes, so decode to a string.
        iqrs_state_b64 = base64.b64encode(iqrs_state_bytes).decode('utf8')

        return make_response_json("Success",
                                  sid=sid,
                                  state_b64=iqrs_state_b64), 200

    # PUT /state
    def set_iqr_state(self):
        """ [See api.rst]
        Set the IQR session state for a given session ID.

        We expect the input bytes to have been generated by the matching
        get-state endpoint (see above). However, unlike the other endpoint's
        return format, the byte input to this endpoint must be encoded in
        URL-safe base64.

        Form Args:
            sid
                Session ID to set the input state to.
            state_base64
                Base64 of the state to set the session to.  This should be
                retrieved from the [GET] /state endpoint.

        Possible error code returns:
            400
                - No session ID provided.
                - No base64 bytes provided.
            404
                No session for the given ID.

        Success returns 200: {
            message = "Success"
            ...
            sid = <str>
        }

        """
        sid = flask.request.form.get('sid', None)
        state_base64 = flask.request.form.get('state_base64', None)

        if sid is None:
            return make_response_json("No session id (sid) provided."), 400
        elif state_base64 is None or len(state_base64) == 0:
            return make_response_json("No state package base64 provided."), 400

        # TODO: Limit the size of input state object? Is this already handled
        #       by other security measures?

        # Encoding is required because the b64decode does not handle being
        # given unicode (python2) or str (python3): needs bytes.

        try:
            # Using urlsafe version because it handles both regular and urlsafe
            # alphabets.
            state_bytes = \
                base64.urlsafe_b64decode(state_base64.encode('utf-8'))
        except (TypeError, binascii.Error) as ex:
            return make_response_json("Invalid base64 input: %s" % str(ex)), \
                   400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            iqrs.set_state_bytes(state_bytes, self.descriptor_factory)
        finally:
            iqrs.lock.release()

        return make_response_json("Success", sid=sid), 200
