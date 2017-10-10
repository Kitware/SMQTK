import base64
import json
import time
import traceback
import uuid
import zipfile

import flask
import six
from io import BytesIO

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
                        "Descriptor generation algorithm to use when requested "
                        "to describe data.",
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

    def describe_base64_data(self, b64, content_type):
        """
        Compute and return the descriptor element for the given base64 data.

        The given data bytes are not retained.

        :param b64: Base64 data string.
        :type b64: str

        :param content_type: Data content type.
        :type content_type: str

        :return: Computed descriptor element.
        :rtype: smqtk.representation.DescriptorElement
        """
        de = DataMemoryElement.from_base64(b64, content_type)
        return self.descriptor_generator.compute_descriptor(
            de, self.descriptor_factory
        )

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

        Arguments:
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
                The base64 byes of the data. This should use the standard and
                URL-safe alphabet as the python ``base64.urlsafe_b64decode``
                module function would expect.
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
                The base64 byes of the data. This should use the standard and
                URL-safe alphabet as the python ``base64.urlsafe_b64decode``
                module function would expect.
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
            try:
                pos_d = set(
                    self.descriptor_index.get_many_descriptors(pos_uuids)
                )
                neg_d = set(
                    self.descriptor_index.get_many_descriptors(neg_uuids)
                )
                neu_d = set(
                    self.descriptor_index.get_many_descriptors(neu_uuids)
                )
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
            num_results = (iqrs.results and len(iqrs.results)) or 0

            if i is None:
                i = 0
            if j is None:
                j = num_results

            try:
                i = int(i)
                j = int(j)
            except ValueError:
                return make_response_json("Invalid bounds index value(s)"), 400

            r = []
            if iqrs.results:
                r = [[d.uuid(), prob] for d, prob in iqrs.ordered_results()[i:j]]

        finally:
            iqrs.lock.release()

        return make_response_json("Returning result pairs",
                                  i=i, j=j, total_results=num_results,
                                  results=r,
                                  sid=sid), 200

    # GET
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
            if not iqrs.positive_descriptors:
                return make_response_json(
                    "No positive labels in current session. Required for a "
                    "supervised classifier.",
                    sid=sid
                ), 400
            if not iqrs.negative_descriptors:
                return make_response_json(
                    "No negative labels in current session. Required for a "
                    "supervised classifier.",
                    sid=sid
                ), 400

            # Get descriptor elements for classification
            try:
                descriptors = list(self.descriptor_index
                                       .get_many_descriptors(uuids))
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
                    {pos_label: iqrs.positive_descriptors,
                     neg_label: iqrs.negative_descriptors}
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
        """
        Create a binary package of a session's IQR state.

        This state is composed of the descriptor vectors, and their UUIDs, that
        were adjudicated positive and negative.

        This function returns a JSON response with the bytes.

        URI Arguments:
            sid
                Session UUID to get the state of.

        This function returns the bytes of the state object (zipfile of json
        dump)

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

        # NOTE: May have to return base64 here instead of bytes.
        return iqrs_state_bytes, 200

    # PUT /state
    def set_iqr_state(self):
        """
        Set the IQR session state for a given session ID.

        We expect to be given a the URL-safe base64 encoding of bytes of the
        zip-file buffer returned from the above ``get_iqr_state`` function.

        """
        sid = flask.request.form.get('sid', None)
        state_base64 = flask.request.form.get('state_base64', None)

        if sid is None:
            return make_response_json("No session id (sid) provided."), 400
        elif state_base64 is None or len(state_base64) == 0:
            return make_response_json("No state package base64 provided."), 400

        # TODO: Limit the size of input state object? Is this already handled by
        #       other security measures?

        # Encoding is required because the b64decode does not handle being
        # given unicode (python2) or str (python3).
        state_bytes = base64.urlsafe_b64decode(state_base64.encode('utf-8'))

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

    # GET /is_ready
    # noinspection PyMethodMayBeStatic
    def is_ready(self):
        """
        Simple function that returns True, indicating that the server is active.
        """
        return make_response_json("Yes, I'm alive."), 200

    def run(self, host=None, port=None, debug=False, **options):
        # Setup REST API here, register methods
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
        self.add_url_rule('/is_ready',
                          view_func=self.is_ready,
                          methods=['GET'])

        super(IqrService, self).run(host, port, debug, **options)
