"""
Video Search blueprint
"""

import base64
import flask
import json
import logging
import multiprocessing
import os
import os.path as osp
# noinspection PyPackageRequirements
import PIL.Image
import random

from smqtk.iqr import IqrController, IqrSessionFusion
from smqtk.utils.configuration import FusionConfiguration
from smqtk.utils.preview_cache import PreviewCache
from smqtk.web.search_app.modules.file_upload import FileUploadMod


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class IQRSearchFusion (flask.Blueprint):

    def __init__(self, name, parent_app, data_set,
                 fusion_setting,
                 url_prefix=None):
        """
        Initialize a generic IQR Search module that uses descriptor/index
        fusion.

        :param name: Name of this blueprint instance
        :type name: str

        :param parent_app: Parent containing flask app instance
        :type parent_app: smqtk.web.search_app.app.search_app

        :param data_set: Data set to work over
        :type data_set: SMQTK.data_rep.DataSet

        :param fusion_setting: Fusion setting to use.
        :type fusion_setting: str

        :param url_prefix: Web address prefix for this blueprint.
        :type url_prefix: str

        :raises ValueError: Invalid Descriptor or indexer type

        """
        super(IQRSearchFusion, self).__init__(
            name, import_name=__name__,
            static_folder=os.path.join(SCRIPT_DIR, "static"),
            template_folder=os.path.join(SCRIPT_DIR, "templates"),
            url_prefix=url_prefix
        )

        # Make sure that the configured descriptor/indexer types exist, as
        # we as their system configuration sections
        if fusion_setting not in FusionConfiguration.available_labels():
            raise ValueError("'%s' not a valid fusion setting."
                             % fusion_setting)

        self._parent_app = parent_app
        self._data_set = data_set
        self._fusion_setting = fusion_setting

        self._explicit_uids = set()
        self._explicit_uids_lock = multiprocessing.RLock()
        # TODO: Read in dict from save file

        # Uploader Sub-Module
        self.upload_work_dir = os.path.join(self.work_dir, "uploads")
        self.mod_upload = FileUploadMod('%s_uploader' % self.name, parent_app,
                                        self.upload_work_dir,
                                        url_prefix='/uploader')
        self.register_blueprint(self.mod_upload)

        # IQR Session control
        # TODO: Move session management to database. Create web-specific
        #       IqrSession class that stores/gets its state directly from
        #       database.
        self._iqr_controller = IqrController()

        # structures for session ingest progress
        # Two levels: SID -> FID
        self._ingest_progress_locks = {}
        self._ingest_progress = {}

        # Preview Image Caching
        # TODO: Initialize this into static directory that is being served.
        self._preview_cache = PreviewCache(osp.join(self.work_dir, "Previews"))

        #
        # Routing
        #

        @self.route("/")
        @self._parent_app.module_login.login_required
        def index():
            r = {
                "module_name": self.name,
                "uploader_url": self.mod_upload.url_prefix,
                "uploader_post_url": self.mod_upload.upload_post_url(),
            }
            r.update(parent_app.nav_bar_content())
            # noinspection PyUnresolvedReferences
            return flask.render_template("iqr_search_index.html", **r)

        @self.route('/iqr_session_info', methods=["GET"])
        @self._parent_app.module_login.login_required
        def iqr_session_info():
            """
            Get information about the current IRQ session
            """
            with self.get_current_iqr_session() as iqrs:
                # noinspection PyProtectedMember
                return flask.jsonify({
                    "uuid": iqrs.uuid,
                    "positive_uids": tuple(iqrs.positive_ids),
                    "negative_uids": tuple(iqrs.negative_ids),
                    "extension_ingest_contents":
                        dict((uid, str(df))
                             for uid, df in iqrs.extension_ds.iteritems()),
                    "FeatureMemory": {
                    }
                })

        @self.route("/check_current_iqr_session")
        @self._parent_app.module_login.login_required
        def check_current_iqr_session():
            """
            Check that the current IQR session exists and is initialized.

            :rtype: {
                    success: bool
                }
            """
            # Getting the current IQR session ensures that one has been
            # constructed for the current session.
            with self.get_current_iqr_session():
                return flask.jsonify({
                    "success": True
                })

        @self.route('/iqr_ingest_file', methods=['POST'])
        @self._parent_app.module_login.login_required
        def iqr_ingest_file():
            """
            Ingest the file with the given UID, getting the path from the
            uploader.

            :return: status message
            :rtype: str

            """
            iqr_sess = self.get_current_iqr_session()
            # TODO: Add status dict with a "GET" method branch for getting that
            #       status information.

            # Start the ingest of a FID when POST
            if flask.request.method == "POST":
                fid = flask.request.form['fid']

                self.log.debug("[%s::%s] Getting temporary filepath from "
                               "uploader module", iqr_sess.uuid, fid)
                upload_filepath = self.mod_upload.get_path_for_id(fid)
                self.mod_upload.clear_completed(fid)

                # Extend session ingest -- modifying
                with iqr_sess:
                    self.log.debug("[%s::%s] Adding new file to extension "
                                   "ingest", iqr_sess.uuid, fid)
                    old_max_uid = iqr_sess.extension_ds.max_uid()
                    upload_data = iqr_sess.extension_ds.add_data_file(upload_filepath)
                    os.remove(upload_filepath)
                    new_max_uid = iqr_sess.extension_ds.max_uid()
                    if old_max_uid == new_max_uid:
                        # re-mark as a positive
                        iqr_sess.adjudicate((upload_data.uid,))
                        return "Already Ingested"

                with iqr_sess:
                    # Compute feature for data -- non-modifying
                    self.log.debug("[%s::%s] Computing feature for file",
                                   iqr_sess.uuid, fid)
                    iqr_sess.reactor.extend(upload_data)
                    iqr_sess.adjudicate((upload_data.uid,))

                return "Finished Ingestion"

        @self.route("/adjudicate", methods=["POST", "GET"])
        @self._parent_app.module_login.login_required
        def adjudicate():
            """
            Update adjudication for this session

            :return: {
                    success: <bool>,
                    message: <str>
                }
            """
            if flask.request.method == "POST":
                fetch = flask.request.form
            elif flask.request.method == "GET":
                fetch = flask.request.args
            else:
                raise RuntimeError("Invalid request method '%s'"
                                   % flask.request.method)

            pos_to_add = json.loads(fetch.get('add_pos', '[]'))
            pos_to_remove = json.loads(fetch.get('remove_pos', '[]'))
            neg_to_add = json.loads(fetch.get('add_neg', '[]'))
            neg_to_remove = json.loads(fetch.get('remove_neg', '[]'))

            self.log.debug("Adjudicated Positive{+%s, -%s}, Negative{+%s, -%s} "
                           % (pos_to_add, pos_to_remove,
                              neg_to_add, neg_to_remove))

            with self.get_current_iqr_session() as iqrs:
                iqrs.adjudicate(pos_to_add, neg_to_add,
                                pos_to_remove, neg_to_remove)
            return flask.jsonify({
                "success": True,
                "message": "Adjudicated Positive{+%s, -%s}, Negative{+%s, -%s} "
                           % (pos_to_add, pos_to_remove,
                              neg_to_add, neg_to_remove)
            })

        @self.route("/get_item_adjudication", methods=["GET"])
        @self._parent_app.module_login.login_required
        def get_adjudication():
            """
            Get the adjudication status of a particular result by ingest ID.

            This should only ever return a dict where one of the two, or
            neither, are labeled True.

            :return: {
                    is_pos: <bool>,
                    is_neg: <bool>
                }
            """
            ingest_uid = flask.request.args['uid']
            with self.get_current_iqr_session() as iqrs:
                return flask.jsonify({
                    "is_pos": ingest_uid in iqrs.positive_ids,
                    "is_neg": ingest_uid in iqrs.negative_ids
                })

        @self.route("/get_positive_uids", methods=["GET"])
        @self._parent_app.module_login.login_required
        def get_positive_uids():
            """
            Get a list of the positive ingest UIDs

            :return: {
                    uids: list of <int>
                }
            """
            with self.get_current_iqr_session() as iqrs:
                return flask.jsonify({
                    "uids": list(iqrs.positive_ids)
                })

        @self.route("/get_random_uids")
        @self._parent_app.module_login.login_required
        def get_random_uids():
            """
            Return to the client a list of all known dataset IDs but in a random
            order. If there is currently an active IQR session with elements in
            its extension ingest, then those IDs are included in the random
            list.

            :return: {
                    uids: list of int
                }
            """
            all_ids = self._data_set.uuids()
            with self.get_current_iqr_session() as iqrs:
                all_ids.update(iqrs.extension_ds.uuids())
            all_ids = list(all_ids)
            random.shuffle(all_ids)
            return flask.jsonify({
                "uids": all_ids
            })

        @self.route("/get_ingest_image_preview_data", methods=["GET"])
        @self._parent_app.module_login.login_required
        def get_ingest_item_image_rep():
            """
            Return the base64 preview image data for the data file associated
            with the give UID.
            """
            uid = flask.request.args['uid']
            info = {
                "success": True,
                "message": None,
                "is_explicit": None,
                "shape": None,  # (width, height)
                "data": None,
                "ext": None,
            }

            de = None
            if self._data_set.has_uuid(uid):
                de = self._data_set.get_data(uid)
                with self._explicit_uids_lock:
                    info["is_explicit"] = uid in self._explicit_uids
            else:
                with self.get_current_iqr_session() as iqrs:
                    if iqrs.extension_ds.has_uid(uid):
                        de = iqrs.extension_ds.get_data(uid)
                        info["is_explicit"] = uid in self._explicit_uids

            if not de:
                info["success"] = False
                info["message"] = "UUID not part of the active data set!"
            else:
                # TODO: Have data-file return an HTML chunk for implementation
                #       defined visualization?
                img_path = self._preview_cache.get_preview_image(de)
                img = PIL.Image.open(img_path)
                info["shape"] = img.size
                with open(img_path, 'rb') as img_file:
                    info["data"] = base64.encodestring(img_file.read())
                info["ext"] = osp.splitext(img_path)[1].lstrip('.')

            return flask.jsonify(info)

        @self.route("/mark_uid_explicit", methods=["POST"])
        @self._parent_app.module_login.login_required
        def mark_uid_explicit():
            """
            Mark a given UID as explicit in its containing ingest.

            :return: Success value of True if the given UID was valid and set
                as explicit in its containing ingest.
            :rtype: {
                "success": bool
            }
            """
            uid = flask.request.form['uid']
            self._explicit_uids.add(uid)
            # TODO: Save out dict

            return flask.jsonify({'success': True})

        @self.route("/iqr_refine", methods=["POST"])
        @self._parent_app.module_login.login_required
        def iqr_refine():
            """
            Classify current IQR session indexer, updating ranking for
            display.

            Fails gracefully if there are no positive[/negative] adjudications.

            Expected Args:
            """
            pos_to_add = json.loads(flask.request.form.get('add_pos', '[]'))
            pos_to_remove = json.loads(flask.request.form.get('remove_pos', '[]'))
            neg_to_add = json.loads(flask.request.form.get('add_neg', '[]'))
            neg_to_remove = json.loads(flask.request.form.get('remove_neg', '[]'))

            with self.get_current_iqr_session() as iqrs:
                try:
                    iqrs.refine(pos_to_add, neg_to_add,
                                pos_to_remove, neg_to_remove)
                    return flask.jsonify({
                        "success": True,
                        "message": "Completed refinement"
                    })
                except Exception, ex:
                    return flask.jsonify({
                        "success": False,
                        "message": "ERROR: " + str(ex)
                    })

        @self.route("/iqr_ordered_results", methods=['GET'])
        @self._parent_app.module_login.login_required
        def get_ordered_results():
            """
            Get ordered (UID, probability) pairs in between the given indices,
            [i, j). If j Is beyond the end of available results, only available
            results are returned.

            This may be empty if no refinement has yet occurred.

            Return format:
            {
                results: [ (uid, probability), ... ]
            }
            """
            with self.get_current_iqr_session() as iqrs:
                i = int(flask.request.args.get('i', 0))
                j = int(flask.request.args.get('j', len(iqrs.results)
                                               if iqrs.results else 0))
                return flask.jsonify({
                    "results": (iqrs.ordered_results or [])[i:j]
                })

        @self.route("/reset_iqr_session", methods=["POST"])
        @self._parent_app.module_login.login_required
        def reset_iqr_session():
            """
            Reset the current IQR session
            """
            with self.get_current_iqr_session() as iqrs:
                iqrs.reset()
                return flask.jsonify({
                    "success": True
                })

    def register_blueprint(self, blueprint, **options):
        """ Add sub-blueprint to a blueprint. """
        def deferred(state):
            if blueprint.url_prefix:
                blueprint.url_prefix = self.url_prefix + blueprint.url_prefix
            else:
                blueprint.url_prefix = self.url_prefix
            state.app.register_blueprint(blueprint, **options)

        self.record(deferred)

    @property
    def log(self):
        return logging.getLogger("smqtk.IQRSearch(%s)" % self.name)

    @property
    def work_dir(self):
        """
        :return: Common work directory for this instance.
        :rtype: str
        """
        return osp.join(self._parent_app.config['WORK_DIR'], "Web", "IQR",
                        self.name)

    def get_current_iqr_session(self):
        """
        Get the current IQR Session instance.

        :return: IqrSession instance for the current session.
        :rtype: smqtk.IQR.iqr_session_fusion.IqrSessionFusion

        """
        with self._iqr_controller:
            sid = flask.session.sid
            if not self._iqr_controller.has_session_uuid(sid):
                sid_work_dir = osp.join(self.work_dir, sid)

                reactor = FusionConfiguration.new_inst(self._fusion_setting)

                iqr_sess = IqrSessionFusion(sid_work_dir, reactor, sid)
                self._iqr_controller.add_session(iqr_sess, sid)
                # If there are things already in our extension ingest, extend
                # the base indexer
                self.log.debug("Extending reactor with existing online ingest "
                               "content")
                reactor.extend(*tuple(iqr_sess.extension_ds))

            return self._iqr_controller.get_session(sid)
