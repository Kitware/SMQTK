import json
import time
import traceback
import uuid

import flask

# import smqtk.algorithms
from smqtk.algorithms import (
    get_classifier_impls,
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
    get_descriptor_index_impls,
)
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

    # POST
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

    # PUT
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

        iqrs.reset()
        self.session_classifiers[sid] = None
        self.session_classifier_dirty[sid] = True
        iqrs.lock.release()

        return make_response_json("Reset IQR session '%s'" % sid,
                                  sid=sid), 200

    # DELETE
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

    # PUT
    def adjudicate(self):
        """
        Incrementally update internal adjudication state given new positives
        and negatives, and optionally IDs for descriptors now marked neutral.

        If the same UUID is present in both positive and negative sets, they
        cancel each other out (remains neutral).

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
        elif pos_uuids is None:
            return make_response_json("No positive UUIDs given"), 400

        pos_uuids = set(json.loads(pos_uuids))
        neg_uuids = set(json.loads(neg_uuids))
        neu_uuids = set(json.loads(neu_uuids))

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

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

        iqrs.lock.release()
        return make_response_json(
            "Finished adjudication",
            sid=sid,
        ), 200

    # PUT
    def refine(self):
        """
        Create or update the session's working index as necessary, ranking
        content by order of relevance.

        Positive and negative UUIDs must be specified as a JSON list. This
        means that string UUIDs must be quoted.

        Form args:
            sid
                UUID of the session to use
            pos_uuids
                list of positive example descriptor UUIDs
            neg_uuids
                list of negative example descriptor UUIDs

        """
        sid = flask.request.form.get('sid', None)
        pos_uuids = flask.request.form.get('pos_uuids', None)
        neg_uuids = flask.request.form.get('neg_uuids', '[]')

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400
        elif pos_uuids is None:
            return make_response_json("No positive UUIDs given"), 400

        pos_uuids = json.loads(pos_uuids)
        neg_uuids = json.loads(neg_uuids)

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id %s not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        # Get appropriate descriptor elements from index for
        # setting new adjudication state.
        try:
            pos_descrs = set(
                self.descriptor_index.get_many_descriptors(pos_uuids)
            )
            neg_descrs = set(
                self.descriptor_index.get_many_descriptors(neg_uuids)
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

        # if a new classifier should be made upon the next
        # classification request.
        diff_pos = \
            pos_descrs.symmetric_difference(
                iqrs.positive_descriptors)
        diff_neg = \
            neg_descrs.symmetric_difference(
                iqrs.negative_descriptors)
        if diff_pos or diff_neg:
            self._log.debug("[%s] session Classifier dirty", sid)
            self.session_classifier_dirty[sid] = True

        self._log.info("[%s] Setting adjudications", sid)
        iqrs.positive_descriptors = pos_descrs
        iqrs.negative_descriptors = neg_descrs

        self._log.info("[%s] Updating working index", sid)
        iqrs.update_working_index(self.neighbor_index)

        self._log.info("[%s] Refining", sid)
        iqrs.refine()

        iqrs.lock.release()
        return make_response_json("Refine complete", sid=sid), 201

    # GET
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

        if iqrs.results:
            size = len(iqrs.results)
        else:
            size = 0

        iqrs.lock.release()
        return make_response_json("Currently %d results for session %s"
                                  % (size, sid),
                                  num_results=size,
                                  sid=sid), 200

    # GET
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
            r = iqrs.ordered_results()[i:j]
            r = [[d.uuid(), v] for d, v in r]

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
                List of descriptor UUIDs to classify. Return list of results
                will be in the same order as this list.

        """
        # Record clean/dirty status after making classifier/refining so we
        # don't train a new classifier when we don't have to.
        sid = flask.request.args.get('sid', None)
        uuids = flask.request.args.get('uuids', None)

        try:
            uuids = json.loads(uuids)
        except ValueError:
            return make_response_json(
                "Failed to decode uuids as json. Given '%s'"
                % uuids
            ), 400

        if sid is None:
            return make_response_json("No session id (sid) provided"), 400

        if not uuids:
            return make_response_json(
                "No descriptor UUIDs provided for classification",
                sid=sid,
            ), 400

        with self.controller:
            if not self.controller.has_session_uuid(sid):
                return make_response_json("session id '%s' not found" % sid,
                                          sid=sid), 404
            iqrs = self.controller.get_session(sid)
            iqrs.lock.acquire()  # lock BEFORE releasing controller

        if not iqrs.positive_descriptors:
            return make_response_json(
                "No positive labels in current session",
                sid=sid
            ), 400
        if not iqrs.negative_descriptors:
            return make_response_json(
                "No negative labels in current session",
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
                            "refine state")

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

        iqrs.lock.release()
        return make_response_json(
            "Finished classification",
            sid=sid,
            uuids=o_uuids,
            proba=o_proba,
        ), 200

    def run(self, host=None, port=None, debug=False, **options):
        # Setup REST API here, register methods
        self.add_url_rule('/session',
                          view_func=self.init_session,
                          methods=['POST'])
        self.add_url_rule('/session',
                          view_func=self.reset_session,
                          methods=['PUT'])
        self.add_url_rule('/session',
                          view_func=self.clean_session,
                          methods=['DELETE'])
        self.add_url_rule('/adjudicate',
                          view_func=self.adjudicate,
                          methods=['PUT'])
        self.add_url_rule('/refine',
                          view_func=self.refine,
                          methods=['PUT'])
        self.add_url_rule('/num_results',
                          view_func=self.num_results,
                          methods=['GET'])
        self.add_url_rule('/get_results',
                          view_func=self.get_results,
                          methods=['GET'])
        self.add_url_rule('/classify',
                          view_func=self.classify,
                          methods=['GET'])

        super(IqrService, self).run(host, port, debug, **options)
