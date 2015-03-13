"""
Video Search blueprint
"""

import base64
import flask
import json
import logging
import os
import os.path as osp
import PIL.Image
import random

from SMQTK.FeatureDescriptors import get_descriptors
from SMQTK.Classifiers import get_classifiers

from SMQTK.IQR import IqrController, IqrSession

from SMQTK.Web.common_flask_blueprints.file_upload import FileUploadMod


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class IQRSearch (flask.Blueprint):

    def __init__(self, name, parent_app, ingest,
                 descriptor_type, classifier_type,
                 url_prefix=None):
        """
        Initialize a generic IQR Search module with a single descriptor and
        classifier.

        :param name: Name of this blueprint instance
        :type name:

        :param parent_app: Parent containing flask app instance
        :type parent_app: SMQTK.Web.SMQTKSearchApp.app.SMQTKSearchApp

        :param ingest: The primary data ingest to search over.
        :type ingest: SMQTK.utils.DataIngest.DataIngest

        :param descriptor_type: Feature Descriptor type string
        :type descriptor_type: str

        :param classifier_type: Classifier type string
        :type classifier_type: str

        :param url_prefix:
        :type url_prefix:

        :raises ValueError: Invalid Descriptor or Classifier type

        """
        super(IQRSearch, self).__init__(
            name, import_name=__name__,
            static_folder=os.path.join(SCRIPT_DIR, "static"),
            template_folder=os.path.join(SCRIPT_DIR, "templates"),
            url_prefix=url_prefix
        )

        # Make sure that the configured descriptor/classifier types exist, as
        # we as their system configuration sections
        if descriptor_type not in get_descriptors():
            raise ValueError("Not a valid descriptor type: %s" % descriptor_type)
        if classifier_type not in get_classifiers():
            raise ValueError("Not a valid classifier type: %s" % classifier_type)
        try:
            parent_app.config['SYSTEM_CONFIG']\
                ['FeatureDescriptors'][descriptor_type]
        except KeyError:
            raise ValueError("No configuration section for descriptor type '%s'"
                             % descriptor_type)
        try:
            parent_app.config['SYSTEM_CONFIG']\
                ['Classifiers'][classifier_type][descriptor_type]\
                ['data_directory']
        except KeyError:
            raise ValueError("No configuration section for classifier type "
                             "'%s' for descriptor '%s'"
                             % (classifier_type, descriptor_type))

        self._parent_app = parent_app
        self._ingest = ingest
        self._fd_type_str = descriptor_type
        self._cl_type_str = classifier_type

        # Uploader Sub-Module
        self.upload_work_dir = os.path.join(
            parent_app.config['WORK_DIR'], "IQR", self.name, "uploads"
        )
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

        #
        # Routing
        #

        @self.route("/")
        @self._parent_app.module_login.login_required
        def index():
            return flask.render_template("iqr_search_index.html", **{
                "module_name": self.name,
                "uploader_url": self.mod_upload.url_prefix,
                "uploader_post_url": self.mod_upload.upload_post_url()
            })

        @self.route('/iqr_session_info', methods=["GET"])
        @self._parent_app.module_login.login_required
        def iqr_session_info():
            """
            Get information about the current IRQ session
            """
            with self.get_current_iqr_session() as iqrs:
                return flask.jsonify({
                    "uuid": iqrs.uuid,
                    "positive_uids": tuple(iqrs.positive_ids),
                    "negative_uids": tuple(iqrs.negative_ids),
                    "extension_ingest_contents":
                        dict((id, str(df))
                             for id, df in iqrs.extension_ingest.iteritems())
                })

        @self.route('/iqr_ingest_file', methods=['POST'])
        @self._parent_app.module_login.login_required
        def iqr_ingest_file():
            """
            Ingest the file with the given UID, getting the path from the
            uploader.

            :return: status message
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
                    old_max_uid = iqr_sess.extension_ingest.max_uid()
                    upload_data = iqr_sess.extension_ingest.add_data_file(upload_filepath)
                    os.remove(upload_filepath)
                    new_max_uid = iqr_sess.extension_ingest.max_uid()
                    if old_max_uid == new_max_uid:
                        return "Already Ingested"

                # Compute feature for data -- non-modifying
                self.log.debug("[%s::%s] Computing feature for file",
                               iqr_sess.uuid, fid)
                feat = iqr_sess.descriptor.compute_feature(upload_data)

                # Extend classifier model with feature data -- modifying
                with iqr_sess:
                    self.log.debug("[%s::%s] Extending classifier model with "
                                   "feature", iqr_sess.uuid, fid)
                    iqr_sess.classifier.extend_model({upload_data.uid: feat})

                return "Finished Ingestion"

        @self.route("/adjudicate", methods=["POST"])
        @self._parent_app.module_login.login_required
        def adjudicate():
            """
            Update adjudication for this session

            :return: {
                    success: <bool>,
                    message: <str>
                }
            """
            pos_to_add = json.loads(flask.request.form.get('add_pos', '[]'))
            pos_to_remove = json.loads(flask.request.form.get('remove_pos', '[]'))
            neg_to_add = json.loads(flask.request.form.get('add_neg', '[]'))
            neg_to_remove = json.loads(flask.request.form.get('remove_neg', '[]'))

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
            ingest_uid = int(flask.request.args['uid'])
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
            all_ids = self._ingest.uids()
            with self.get_current_iqr_session() as iqrs:
                all_ids.extend(iqrs.extension_ingest.uids())
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
            uid = int(flask.request.args['uid'])
            info = {
                "success": True,
                "message": None,
                "is_explicit": None,
                "shape": None,
                "data": None,
                "ext": None,
            }

            df = None
            if uid in self._ingest.has_uid(uid):
                df = self._ingest.get_data(uid)
                info["is_explicit"] = self._ingest.is_explicit(uid)
            else:
                with self.get_current_iqr_session() as iqrs:
                    if iqrs.extension_ingest.has_uid(uid):
                        df = iqrs.extension_ingest.get_data(uid)
                        info["is_explicit"] = iqrs.extension_ingest.is_explicit(uid)

            if not df:
                info["success"] = False
                info["message"] = "UID not part of the ingest"
            else:
                img_path = df.get_preview_image()
                img = PIL.Image.open(img_path)
                info["shape"] = img.size
                with open(img_path, 'rb').read() as img_data:
                    info["data"] = base64.encodestring(img_data)
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
            uid = int(flask.request.form['uid'])
            if self._ingest.has_uid(uid):
                self._ingest.set_explicit(uid)
            else:
                with self.get_current_iqr_session() as iqrs:
                    if iqrs.extension_ingest.has_uid(uid):
                        iqrs.extension_ingest.set_explicit(uid)

            return flask.jsonify({'r': True})

        @self.route("/iqr_refine", methods=["POST"])
        @self._parent_app.module_login.login_required
        def iqr_refine():
            """
            Classify current IQR session classifier, updating ranking for
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
                        "message": str(ex)
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
        return logging.getLogger("IqrSearch(%s)" % self.name)

    def get_current_iqr_session(self):
        """
        Get the current IQR Session instance.

        :rtype: SMQTK.IQR.iqr_session.IqrSession

        """
        with self._iqr_controller:
            sid = flask.session.sid
            if not self._iqr_controller.has_session_uuid(sid):
                sid_work_dir = osp.join(self._parent_app.config['WORK_DIR'],
                                        "IQR", self.name, sid)
                descriptor = get_descriptors()[self._fd_type_str](
                    osp.join(
                        self._parent_app.config['DATA_DIR'],
                        self._parent_app.config['SYSTEM_CONFIG']\
                            ['FeatureDescriptors'][self._fd_type_str]\
                            ['data_directory']
                    ),
                    osp.join(sid_work_dir, 'fd')
                )
                classifier = get_classifiers()[self._cl_type_str](
                    osp.join(
                        self._parent_app.config['DATA_DIR'],
                        self._parent_app.config['SYSTEM_CONFIG']\
                            ['Classifiers'][self._cl_type_str]\
                            [self._fd_type_str]['data_directory']
                    ),
                    osp.join(sid_work_dir, 'cl'),
                )
                online_ingest = self._ingest.__class__(
                    osp.join(sid_work_dir, 'online-ingest'),
                    osp.join(sid_work_dir, 'online-ingest-work'),
                    starting_index=self._ingest.max_uid() + 1
                )
                iqr_sess = IqrSession(sid_work_dir, descriptor, classifier,
                                      online_ingest, sid)
                self._iqr_controller.add_session(iqr_sess, sid)

            return self._iqr_controller.get_session(sid)
