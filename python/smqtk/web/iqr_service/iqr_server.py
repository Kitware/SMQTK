import base64
import binascii
import collections
import json
import multiprocessing
import time
import traceback
import uuid

import flask

# import smqtk.algorithms
from smqtk.algorithms import (
    get_classifier_impls,
    get_descriptor_generator_impls,
    get_nn_index_impls,
    get_relevancy_index_impls,
    SupervisedClassifier,
)
from smqtk.iqr import (
    iqr_controller,
    iqr_session,
)
from smqtk.representation import (
    ClassificationElementFactory,
    DescriptorElementFactory,
    get_descriptor_index_impls,
)
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.utils import (
    merge_dict,
    plugin,
)
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
    :rtype: list[collections.Hashable]

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
    elif not all(isinstance(el, collections.Hashable)
                 for el in v_list):
        raise ValueError("Not all JSON list parts were hashable values.")
    return v_list


class IqrService (SmqtkWebApp):
    """
    Configuration Notes
    -------------------
    ``descriptor_index`` will currently be configured twice: once for the
    global index and once for the nearest neighbors index. These will probably
    be the set to the same index. In more detail, the global descriptor index
    is used when the "refine" endpoint is given descriptor UUIDs
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(IqrService, cls).get_default_config()

        c_rel_index = plugin.make_config(
            get_relevancy_index_impls()
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
                    "descriptor_index":
                        "This is the index from which given positive and "
                        "negative example descriptors are retrieved from. "
                        "Not used for nearest neighbor querying. "
                        "This index must contain all descriptors that could "
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
                    "descriptor_generator": plugin.make_config(
                        get_descriptor_generator_impls()
                    ),
                    "descriptor_index": plugin.make_config(
                        get_descriptor_index_impls()
                    ),
                    "neighbor_index":
                        plugin.make_config(get_nn_index_impls()),
                    "classifier_config":
                        plugin.make_config(get_classifier_impls()),
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
        self.descriptor_generator = plugin.from_plugin_config(
            json_config['iqr_service']['plugins']['descriptor_generator'],
            get_descriptor_generator_impls(),
        )

        #: :type: smqtk.representation.DescriptorIndex
        self.descriptor_index = plugin.from_plugin_config(
            json_config['iqr_service']['plugins']['descriptor_index'],
            get_descriptor_index_impls(),
        )

        #: :type: smqtk.algorithms.NearestNeighborsIndex
        self.neighbor_index = plugin.from_plugin_config(
            json_config['iqr_service']['plugins']['neighbor_index'],
            get_nn_index_impls(),
        )
        self.neighbor_index_lock = multiprocessing.RLock()

        self.rel_index_config = \
            json_config['iqr_service']['plugins']['relevancy_index_config']

        # Record of trained classifiers for a session. Session classifier
        # modifications locked under the parent session's global lock.
        #: :type: dict[collections.Hashable, SupervisedClassifier | None]
        self.session_classifiers = {}
        # Control for knowing when a new classifier should be trained for a
        # session (True == train new classifier). Modification for specific
        # sessions under parent session's lock.
        #: :type: dict[collections.Hashable, bool]
        self.session_classifier_dirty = {}

        def session_expire_callback(session):
            """
            :type session: smqtk.iqr.IqrSession
            """
            with session:
                self._log.debug("Removing session %s classifier", session.uuid)
                del self.session_classifiers[session.uuid]
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
                          methods=['POST'])
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
        return self.descriptor_generator.compute_descriptor(
            de, self.descriptor_factory
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
        configured descriptor index.

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
        self.descriptor_index.add_descriptor(descriptor)
        return make_response_json("Success",
                                  uid=descriptor.uuid(),
                                  size=self.descriptor_index.count()), 201

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
                    self.descriptor_index.get_many_descriptors(descr_uid_list)
                self.neighbor_index.update_index(descr_elems)
            except KeyError:
                # Some UIDs are not present in the current index.  Isolate
                # which UIDs are not contained.
                uids_not_ingested = []
                for uid in descr_uid_list:
                    if not self.descriptor_index.has_descriptor(uid):
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
            if str(e) == "Incorrect padding":
                return make_response_json("Failed to parse base64 data."), 400
            # In case some other exception is raised, actually a server error.
            raise

        n_elems, n_dists = self.neighbor_index.nn(descriptor, k)
        return make_response_json("Success",
                                  neighbor_uids=[e.uuid() for e in n_elems],
                                  neighbor_dists=[d for d in n_dists]), 200

    # POST /uid_nearest_neighbors
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

        Form Arguments:
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
        uid = flask.request.form.get('uid', None)
        k_str = flask.request.form.get('k', None)
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
            descriptor = self.descriptor_index.get_descriptor(uid)
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
            wi_count = iqrs.working_index.count()

        finally:
            iqrs.lock.release()

        return make_response_json("Session '%s' info" % sid,
                                  sid=sid,
                                  uuids_pos=uuids_pos,
                                  uuids_neg=uuids_neg,
                                  uuids_pos_ext=uuids_pos_external,
                                  uuids_neg_ext=uuids_neg_external,
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
        configured descriptor index.

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
            self._log.debug("Getting the descriptors for UUIDs")
            # get_many_descriptors can raise KeyError
            pos_d = set(
                self.descriptor_index.get_many_descriptors(pos_uuids)
            )
            neg_d = set(
                self.descriptor_index.get_many_descriptors(neg_uuids)
            )
            neu_d = set(
                self.descriptor_index.get_many_descriptors(neu_uuids)
            )

            # Record previous pos/neg descriptors for determining if an
            # existing classifier is dirty after this adjudication.
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
            self._log.warn(traceback.format_exc())
            return make_response_json(
                "Descriptor UUID '%s' cannot be found in the "
                "configured descriptor index."
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
        Update the working index based on the currently positive examples and
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
            iqrs.update_working_index(self.neighbor_index)
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
        (Re)Create ranking of working index content by order of relevance to
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

        finally:
            iqrs.lock.release()

        return make_response_json("Refine complete", sid=sid), 201

    # GET /num_results
    def num_results(self):
        """
        Get the total number of results in the ranking.

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
        Get the ordered ranking results between two index positions (inclusive,
        exclusive).

        This returns a results slice in the same way that python would handle a
        list slice.

        If the requested session has not been refined yet (no ranking), all
        result requests will be empty.

        URI Args:
            sid
                UUID of the session to use
            i
                Starting index (inclusive)
            j
                Ending index (exclusive)

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
            if not iqrs.results:
                return make_response_json("Returning result pairs",
                                          i=0, j=0, total_results=0,
                                          results=[], sid=sid), 200

            num_results = len(iqrs.results)

            # int() can raise ValueError, catch
            i = 0 if i is None else int(i)
            j = num_results if j is None else int(j)

            # We ensured i, j are valid by this point
            r = [[d.uuid(), prob] for d, prob in iqrs.ordered_results()[i:j]]

        except ValueError:
            return make_response_json("Invalid bounds index value(s)"), 400

        finally:
            iqrs.lock.release()

        return make_response_json("Returning result pairs",
                                  i=i, j=j, total_results=num_results,
                                  results=r,
                                  sid=sid), 200

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
                associate to descriptors in the configured descriptor index.

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
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        try:
            all_pos = (iqrs.external_positive_descriptors |
                       iqrs.positive_descriptors)
            if not all_pos:
                return make_response_json(
                    "No positive labels in current session. Required for a "
                    "supervised classifier.",
                    sid=sid
                ), 400
            all_neg = (iqrs.external_negative_descriptors |
                       iqrs.negative_descriptors)
            if not all_neg:
                return make_response_json(
                    "No negative labels in current session. Required for a "
                    "supervised classifier.",
                    sid=sid
                ), 400

            # Get descriptor elements for classification
            # get_many_descriptors can raise KeyError
            descriptors = list(self.descriptor_index
                               .get_many_descriptors(uuids))

            classifier = self.session_classifiers.get(sid, None)

            pos_label = "positive"
            neg_label = "negative"
            if self.session_classifier_dirty[sid] or classifier is None:
                self._log.debug("Training new classifier for current "
                                "adjudication state...")

                #: :type: SupervisedClassifier
                classifier = plugin.from_plugin_config(
                    self.classifier_config,
                    get_classifier_impls(sub_interface=SupervisedClassifier)
                )
                classifier.train(
                    {pos_label: all_pos,
                     neg_label: all_neg}
                )

                self.session_classifiers[sid] = classifier
                self.session_classifier_dirty[sid] = False

            classifications = classifier.classify_async(
                descriptors, self.classification_factory,
                use_multiprocessing=True, ri=1.0
            )

            # Format output to be parallel lists of UUIDs input and
            # positive class classification scores.
            o_uuids = []
            o_proba = []
            for d in descriptors:
                o_uuids.append(d.uuid())
                o_proba.append(classifications[d][pos_label])

            assert uuids == o_uuids, \
                "Output UUID list is not congruent with INPUT list."

        except KeyError as ex:
            err_uuid = str(ex)
            self._log.warn(traceback.format_exc())
            return make_response_json(
                "Descriptor UUID '%s' cannot be found in the "
                "configured descriptor index."
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
