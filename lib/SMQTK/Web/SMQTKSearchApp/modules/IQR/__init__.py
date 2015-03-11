"""
Video Search blueprint
"""

import flask
import logging
import multiprocessing
import os
import os.path as osp

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
            parent_app.system_config['FeatureDescriptors'][self._fd_type_str]
        except KeyError:
            raise ValueError("No configuration section for descriptor type '%s'"
                             % descriptor_type)
        try:
            parent_app.system_config['Classifiers'][self._cl_type_str] \
                [self._fd_type_str]['data_directory']
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

        @self.route('/iqr_ingest_file', methods=['GET', 'POST'])
        def iqr_ingest_file():
            """
            Ingest the file with the given UID, getting the path from the
            uploader.
            """
            # Start the ingest of a FID when POST
            if flask.request.method == "POST":
                status = {}
                status_lock = multiprocessing.RLock()

                fid = flask.request.form['fid']
                upload_filepath = self.mod_upload.get_path_for_id(fid)
                self.mod_upload.clear_completed(fid)

                # Extend session ingest
                os.remove(upload_filepath)

                # Compute feature for data

                # Extend classifier model with feature data

                return "Return Message"

            # Return ingest status when GET
            else:
                fid = flask.request.args['fid']
                # return flask.jsonify({})
                return None

    def register_blueprint(self, blueprint, **options):
        """ Add sub-blueprint to a blueprint. """
        def deferred(state):
            print "[sub-BP-reg-deferred] Registering new blueprint with URL " \
                  "prefix '%s' underneath parent prefix '%s'" \
                  % (blueprint.url_prefix, self.url_prefix)
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
                    self._parent_app.system_config['FeatureDescriptors']
                                                  [self._fd_type_str]
                                                  ['data_directory'],
                    osp.join(sid_work_dir, 'fd')
                )
                classifier = get_classifiers()[self._cl_type_str](
                    self._parent_app.system_config['Classifiers']
                                                  [self._cl_type_str]
                                                  [self._fd_type_str]
                                                  ['data_directory'],
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
