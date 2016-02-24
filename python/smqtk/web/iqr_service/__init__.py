import json
import time
import traceback
import uuid

import flask

# import smqtk.algorithms
from smqtk.algorithms import (
    get_relevancy_index_impls,
    get_nn_index_impls,
)
from smqtk.iqr import (
    iqr_controller,
    iqr_session,
)
from smqtk.representation import (
    get_descriptor_index_impls,
)
from smqtk.utils import (
    merge_dict,
    plugin,
)
from smqtk.web import SmqtkWebApp


__author__ = "paul.tunison@kitware.com"


def new_uuid():
    return str(uuid.uuid1(clock_seq=int(time.time() * 1000000))).replace('-', '')


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
                "positive_seed_neighbors": 500,

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
                        "positive descriptors from."
                },
                "plugins": {
                    "relevancy_index_config": c_rel_index,
                    "descriptor_index": plugin.make_config(
                        get_descriptor_index_impls()
                    ),
                    "neighbor_index": plugin.make_config(get_nn_index_impls()),
                }
            }
        })
        return c

    def __init__(self, json_config):
        super(IqrService, self).__init__(json_config)

        # Initialize from config
        self.positive_seed_neighbors = \
            json_config['iqr_service']['positive_seed_neighbors']

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

        self.controller = iqr_controller.IqrController()

        # TODO: Timer for removing a session if it hasn't been used for N seconds

    # PUT
    def init_session(self):
        """
        Initialize a new session in the controller.

        Form args:
            session_id
                Optional UUID string specification of session to initialize. If
                one is not provided, we create one here.

        """
        sid = flask.request.form.get('session_id', None)
        if sid is None:
            sid = new_uuid()

        iqrs = iqr_session.IqrSession()
        self.controller.add_session(iqrs, sid)

        return make_response_json("Created new session with ID '%s'" % sid,
                                  sid=sid), 201

    # PUT
    def reset_session(self):
        """
        Reset an existing session. This does not remove the session, so actions
        may

        Form args:
            session_id
                UUID (string) of the session to reset.

        """
        sid = flask.request.form.get('session_id', None)

        if sid is None:
            return make_response_json("No session_id provided"), 400

        try:
            with self.controller.get_session(sid) as iqrs:
                iqrs.reset()
            return make_response_json("Reset IQR session '%s'" % sid,
                                      sid=sid), 200
        except KeyError:
            return make_response_json("session_id '%s' not found" % sid,
                                      sid=sid), 404

    # PUT
    def clean_session(self):
        """
        Clean resources associated with the session of the given UUID.

        Similar to session resetting, but this also removed the session
        resource causing future references without another initialize to return
        errors.

        Form args:
            session_id
                UUID of the session to clean.

        """
        sid = flask.request.form.get('session_id', None)

        if sid is None:
            return make_response_json("No session_id provided"), 400

        try:
            with self.controller:
                with self.controller.get_session(sid) as iqrs:
                    iqrs.reset()
                self.controller.remove_session(sid)
            return make_response_json("Cleaned session resources for '%s'"
                                      % sid,
                                      sid=sid), 200
        except KeyError:
            return make_response_json("session_id '%s' not found" % sid,
                                      sid=sid), 404

    # PUT
    def refine(self):
        """
        Create or update the session's working index as necessary, ranking
        content by order of relevance.

        Positive and negative UUIDs must be specified as a JSON list. This
        means that string UUIDs must be quoted.

        Form args:
            session_id
                UUID of the session to use
            pos_uuids
                list of positive example descriptor UUIDs
            neg_uuids
                list of negative example descriptor UUIDs

        """
        sid = flask.request.form.get('session_id', None)
        pos_uuids = flask.request.form.get('pos_uuids', None)
        neg_uuids = flask.request.form.get('neg_uuids', '[]')

        if sid is None:
            return make_response_json("No session_id provided"), 400
        elif pos_uuids is None:
            return make_response_json("No positive UUIDs given"), 400

        pos_uuids = json.loads(pos_uuids)
        neg_uuids = json.loads(neg_uuids)

        try:
            with self.controller:
                with self.controller.get_session(sid) as iqrs:
                    # Get appropriate descriptor elements from index for
                    # setting new adjudication state.
                    try:
                        pos_descrs = list(
                            self.descriptor_index.get_many_descriptors(*pos_uuids)
                        )
                        neg_descrs = list(
                            self.descriptor_index.get_many_descriptors(*neg_uuids)
                        )
                        iqrs.adjudicate(pos_descrs, neg_descrs)
                        msg = "[set adjudications]"
                    except KeyError, ex:
                        err_uuid = str(ex)
                        self._log.warn(traceback.format_exc())
                        return make_response_json(
                            "Descriptor UUID '%s' cannot be found in the "
                            "configured descriptor index."
                            % err_uuid,
                            sid=sid,
                            uuid=err_uuid,
                        ), 404

                    iqrs.update_working_index(self.neighbor_index)
                    msg += '[updated working index]'

                    iqrs.refine()
                    msg += '[refinement completed]'

            return make_response_json("Steps completed: %s" % msg, sid=sid), 201

        except KeyError:
            return make_response_json("session_id '%s' not found" % sid,
                                      sid=sid), 404

    # GET
    def num_results(self):
        """
        Get the total number of results in the ranking.

        URI Args:
            session_id
                UUID of the session to use

        """
        sid = flask.request.args.get('session_id', None)

        if sid is None:
            return make_response_json("No session_id provided"), 400

        try:
            with self.controller:
                with self.controller.get_session(sid) as iqrs:
                    if iqrs.results:
                        size = len(iqrs.results)
                    else:
                        size = 0
            return make_response_json("Currently %d results for session %s"
                                      % (size, sid),
                                      num_results=size,
                                      sid=sid), 200
        except KeyError:
            return make_response_json("session_id '%s' not found" % sid,
                                      sid=sid), 404

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
            session_id
                UUID of the session to use
            i
                Starting index (inclusive)
            j
                Ending index (exclusive)

        """
        sid = flask.request.args.get('session_id', None)
        i = flask.request.args.get('i', None)
        j = flask.request.args.get('j', None)

        if sid is None:
            return make_response_json("No session_id provided"), 400

        try:
            with self.controller:
                with self.controller.get_session(sid) as iqrs:
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
            return make_response_json("Returning result pairs",
                                      i=i, j=j, total_results=num_results,
                                      results=r,
                                      sid=sid), 200
        except KeyError:
            return make_response_json("session_id '%s' not found" % sid,
                                      sid=sid), 404

    def run(self, host=None, port=None, debug=False, **options):
        # Setup REST API here, register methods
        self.add_url_rule('/init_session',
                          view_func=self.init_session,
                          methods=['POST'])
        self.add_url_rule('/reset_session',
                          view_func=self.reset_session,
                          methods=['PUT'])
        self.add_url_rule('/clean_session',
                          view_func=self.clean_session,
                          methods=['DELETE'])
        self.add_url_rule('/refine',
                          view_func=self.refine,
                          methods=['PUT'])
        self.add_url_rule('/num_results',
                          view_func=self.num_results,
                          methods=['GET'])
        self.add_url_rule('/get_results',
                          view_func=self.get_results,
                          methods=['GET'])

        super(IqrService, self).run(host, port, debug, **options)
