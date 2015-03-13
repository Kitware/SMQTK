# coding=utf-8

import base64
import flask
import json
import logging
import multiprocessing
import os
import os.path as osp
import PIL.Image
import random
from StringIO import StringIO
import tempfile
import uuid

from masir import IngestManager

from masir.search.colordescriptor import \
    ColorDescriptor_CSIFT, ColorDescriptor_TCH
from masir.search.iqr_controller import IqrController
from masir.utils.SimpleBoundingBox import SimpleBoundingBox


script_dir = osp.dirname(osp.abspath(__file__))


class IqrSessionCookieMetadata (object):

    def __init__(self, uid, descriptor_type):
        self.uuid = uid
        self.type = descriptor_type

    def __repr__(self):
        return "IqrSessionCookieMetadata{%s, %s}" \
            % (self.uuid, self.type)


# noinspection PyUnusedLocal
class SearchMod (flask.Blueprint):
    """
    Blueprint module for MASIR searching (combined / separate image and video
    searching)
    """

    # IQR Session metadata key in user's session cookie.
    # Metadata members {
    #   uuid: UUID of the IqrSession
    #   type: Descriptor type used by the current IQR session
    # }
    SESSION_KEY_IQR = 'iqr_session'

    # Mapping of string label to FeatureDescriptor implementation
    #: :type: dict of (str, type)
    FEATURE_DESCRIPTOR_TYPES = {
        "CSIFT": ColorDescriptor_CSIFT,
        "TCH": ColorDescriptor_TCH,
    }
    FEATURE_DESCRIPTOR_TYPE_DEFAULT = "CSIFT"

    RANDOM_SEED = 13579

    @property
    def log(self):
        return logging.getLogger('SearchMod')

    def __init__(self, parent_app):
        """
        Initialize the Search module

        :param parent_app: Parent application that is loading this using this
            module.
        :type parent_app: MasirApp

        """
        super(SearchMod, self).__init__(
            'search', __name__,
            static_folder=osp.join(script_dir, 'static'),
            template_folder=osp.join(script_dir, 'templates')
        )

        self._parent_app = parent_app

        # File chunk aggregation
        #   Top level key is the file ID of the upload. The dictionary
        #   underneath that is the index ID'd chunks. When all chunks are
        #   present, the file is written and the entry in this map is removed.
        #: :type: dict of (str, dict of (int, StringIO))
        self._file_chunks = {}
        # Lock per file ID so as to not collide when uploading multiple chunks
        #: :type: dict of (str, RLock)
        self._fid_locks = {}

        # Session controller
        self._iqr_controller = IqrController()

        self._primary_ingest = IngestManager(parent_app.config['DIR_DATA'])

        #
        # Routing
        #

        @self.route('/')
        @self._parent_app.module_login.login_required
        def search_index():
            """
            MASIR IQR Search main-page
            """
            return flask.render_template("search_index.html")

        @self.route('/_info')
        @self._parent_app.module_login.login_required
        def info():
            """
            Debugging getter, retuning JSON of some current information
            """
            iqr_md = self.get_iqr_session_metadata()
            session_stats = None
            fm_stats = None
            if iqr_md and self._iqr_controller.has_session_uuid(iqr_md.uuid):
                with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                    session_stats = {
                        "positive_ids": tuple(iqr_s.positive_ids),
                        "negative_ids": tuple(iqr_s.negative_ids)
                    }

                    iqr_fm = iqr_s.feature_memory
                    fm_stats = {
                        "num_ids": len(iqr_fm.get_ids()),
                        "num_bg": len(iqr_fm.get_bg_ids()),
                        "feature_mat.shape": iqr_fm.get_feature_matrix().shape,
                        "kernel_mat.shape": iqr_fm.get_kernel_matrix().shape,
                    }

            r = {
                "iqr_md": str(iqr_md),
                "session": dict(flask.session.items()),
                "session_stats": session_stats,
                "controller_session_uids": self._iqr_controller.session_uuids(),
                "iqr_feature_memory": fm_stats,
            }

            self.log.debug("Info data: %s", r)
            return flask.jsonify(r)

        @self.route('/_initialize_new_iqr_session')
        @self._parent_app.module_login.login_required
        def init_new_iqr_session():
            """
            Initialize a new IQR session to a given FeatureDescriptor type,
            setting the session's UUID to the current user's session.

            If no feature descriptor type is specified, we revert to the default
            type (see class definition).

            Expected arguments {
                descriptor_type: String type of the descriptor to use
            }

            JSON return {
                success: Boolean success flag
                message: Return status / error message
                uuid: New IQR session UUID string. Also recorded in session
            }

            """
            descriptor_type_str = flask.request.args.get('descriptor_type',
                                                         None)
            # Default fall-back when none provided
            if descriptor_type_str is None:
                descriptor_type_str = self.FEATURE_DESCRIPTOR_TYPE_DEFAULT

            # Returns
            success = False
            message = None
            uid = None

            try:
                uid = self.iqr_session_init_new(descriptor_type_str)
                success = True
                message = ("Successfully created new IQR session for "
                           "descriptor type '%s'" % descriptor_type_str)
            except RuntimeError, ex:
                message = str(ex)
            except KeyError, ex:
                message = ("Invalid descriptor type '%s'"
                           % descriptor_type_str)
            except Exception, ex:
                message = "Other exception occurred: %s" % str(ex)

            return flask.jsonify({
                'success': success,
                'message': message,
                'uuid': uid
            })

        @self.route('/_iqr_session_remove')
        @self._parent_app.module_login.login_required
        def iqr_session_remove():
            """
            Remove the current IQR Session. If there is no current IQR session,
            this is does nothing.

            JSON return {
                message: return status / error message
            }

            """
            # Returns
            message = None

            try:
                if self.iqr_session_remove():
                    message = 'IQR Session successfully removed'
                else:
                    message = 'Superfluous IQR session ID in cookie ' \
                              'cleaned up'
            except KeyError, ex:
                message = "No IQR Session to remove."
            except Exception, ex:
                message = "Other exception occurred: %s" % str(ex)

            return flask.jsonify({
                'message': message
            })


        @self.route('/_check_iqr_session')
        @self._parent_app.module_login.login_required
        def check_iqr_session():
            """
            Check that the current session has a (valid) IQR Session reference

            Returns JSON object with the following format:
            {
                new: <bool>,  // If the current IQR Session was just created
                success: <bool>,  // Boolean success/failure flag
                message: <string>,  // message describing return state
            }

            """
            # Returns
            new = False
            success = False
            message = "Failed to check current IQR Session status"

            iqr_md = self.get_iqr_session_metadata()
            self.log.debug("Current iqr session metadata: %s", iqr_md)
            if iqr_md is None:
                self.log.debug("Initializing new IQR Session")
                # Construct a new IQR session based on current descriptor setting
                self.iqr_session_init_new(self.FEATURE_DESCRIPTOR_TYPE_DEFAULT),
                iqr_md = self.get_iqr_session_metadata()
                self.log.debug("New session metadata: %s", iqr_md)
                success = new = True
                message = "Initialized new IQR session with feature " \
                          "descriptor 'CSIFT'"
            else:
                self.log.debug("Found existing IQR Session metadata")
                # Current session UUID may associate to a current IQR session
                # or not.
                if iqr_md.uuid not in self._iqr_controller.session_uuids():
                    # Cruft in the current session. Clear and reinitialize
                    self.log.debug("Replacing out of date session and metadata")
                    self.iqr_session_remove()
                    self.iqr_session_init_new(iqr_md.type)
                    flask.session.modified = True
                    success = True
                    message = ("Removed out-of-date metadata and reinitialized "
                               "IQR Session")
                else:
                    # the current metadata is valid
                    self.log.debug("Valid IQR session already initialized")
                    success = True
                    message = "Valid IQR Session already initialized."

            return flask.jsonify({
                'new': new,
                'success': success,
                'message': message,
                'uuid': iqr_md.uuid
            })

        @self.route('/_upload', methods=["POST"])
        @self._parent_app.module_login.login_required
        def upload():
            """
            Handle image/video file uploads
            """
            form = flask.request.form
            self.log.debug("POST form contents: %s" % str(flask.request.form))

            fid = form['flowIdentifier']
            success = True
            message = None
            chunk_size = int(form['flowChunkSize'])
            current_chunk = int(form['flowChunkNumber'])
            total_chunks = int(form['flowTotalChunks'])
            filename = form['flowFilename']

            #: :type: FileStorage
            chunk_data = flask.request.files['file']

            with self._fid_locks.setdefault(fid, multiprocessing.RLock()):
                # Create new entry in chunk map / add to existing entry
                # - Need to explicitly copy the buffered data as the file object
                #   closes between chunk messages.
                self._file_chunks.setdefault(fid, {})[current_chunk] \
                    = StringIO(chunk_data.read())
                message = "Uploaded chunk #%d/%d for file '%s'" \
                    % (current_chunk, total_chunks, filename)

                if total_chunks == len(self._file_chunks[fid]):
                    self.log.debug("[%s] Final chunk uploaded",
                                   filename)
                    fullfile_path = img_files = None
                    # have all chucks in memory now
                    try:
                        # Combine chunks into single file
                        file_ext = osp.splitext(filename)[1]
                        fullfile_path = self._write_file_chunks(
                            self._file_chunks[fid], file_ext)
                        self.log.debug("[%s] saved from chunks: %s",
                                       filename, fullfile_path)
                        # now in file, free up dict memory
                        del self._file_chunks[fid]

                        # May have been given either an image or a video.
                        img_files = self._resolve_input_images(fullfile_path)
                        self.log.debug("[%s] Resolved input imagery: %s",
                                       filename, img_files)

                        # Adding uploaded file to current IQR session
                        self.iqr_session_ingest_files(*img_files)
                        message = "[%s] IQR Session Ingested file" % filename

                    except IOError, ex:
                        self.log.debug("[%s] Failed to extend IQR session "
                                       "model", filename)
                        success = False
                        message = "Failed to write out combined chunks for " \
                                  "file %s: %s" % (filename, str(ex))

                    except NotImplementedError, ex:
                        success = False
                        message = "Encountered non-implemented code path: %s" \
                                  % str(ex)

                    except ValueError, ex:
                        message = "Repeat image upload. Ignoring ingest. " \
                                  "Error: %s" % str(ex)

                    finally:
                        # remove chunk map lock entry and any temp files
                        del self._fid_locks[fid]
                        if fullfile_path:
                            os.remove(fullfile_path)
                        if img_files and img_files[0] != fullfile_path:
                            for f in img_files:
                                os.remove(f)

            # return flask.jsonify({
            #     'id': fid,
            #     'success': success,
            #     'message': message,
            #
            #     'chunk_size': chunk_size,
            #     'current_chunk': current_chunk,
            #     'total_chunks': total_chunks,
            #     'filename': filename,
            # })
            # Flow only displays return as a string, so just returning the
            # message component.
            return message

        @self.route('/_iqr_adjudicate', methods=['GET'])
        @self._parent_app.module_login.login_required
        def iqr_adjudicate():
            """
            Receive clip adjudication messages

            """
            pos_to_add = json.loads(flask.request.args.get('add_pos', '[]'))
            pos_to_remove = json.loads(flask.request.args.get('remove_pos', '[]'))
            neg_to_add = json.loads(flask.request.args.get('add_neg', '[]'))
            neg_to_remove = json.loads(flask.request.args.get('remove_neg', '[]'))

            iqr_md = self.get_iqr_session_metadata()
            with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                iqr_s.adjudicate(pos_to_add, neg_to_add,
                                 pos_to_remove, neg_to_remove)

            return flask.jsonify({
                "success": True,
                "message": "Adjudicated Positive{+%s, -%s}, Negative{+%s, -%s} "
                           % (pos_to_add, pos_to_remove,
                              neg_to_add, neg_to_remove)
            })

        @self.route('/_iqr_refine', methods=['GET'])
        @self._parent_app.module_login.login_required
        def iqr_refine():
            """
            Refine current IQR session, returning when complete.
            """
            success = False
            message = None

            iqr_md = self.get_iqr_session_metadata()
            with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                try:
                    iqr_s.refine()
                    success = True
                    message = "Complete IQR refinement iteration"
                except Exception, ex:
                    message = "Failed IQR refinement iteration: %s" % str(ex)

            return flask.jsonify({
                "success": success,
                "message": message
            })

        @self.route('/_iqr_get_result_adjudication')
        @self._parent_app.module_login.login_required
        def iqr_get_result_adjudication():
            """
            Return adjudication flags for the given result ID.

            A given ID can only ever be positive OR negative OR neither.
            """
            uid = int(flask.request.args['id'])

            is_pos = False
            is_neg = False

            iqr_md = self.get_iqr_session_metadata()
            if iqr_md:
                with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                    is_pos = uid in iqr_s.positive_ids
                    is_neg = uid in iqr_s.negative_ids

            return flask.jsonify({
                "is_pos": is_pos,
                "is_neg": is_neg,
            })

        @self.route('/_iqr_get_results')
        @self._parent_app.module_login.login_required
        def iqr_get_results():
            """
            Get all results in an ordered list.

            Optional geo-filtering bbox parameter: ``geofilter``
            - format: "...?[...]geofilter=x1,y1,x2,y2[&...]"
            - doesn't matter spatially where the two points are (ul, ll, ur, lr)

            If there hasn't been a refinement yet, this returns an empty list.

            Return list element format:
                (uid, probability)

            """
            geofilter_bbox = self._parse_geo_filter_arg(
                flask.request.args.get('geofilter', None)
            )
            self.log.info("Results fetch geo filter: %s", geofilter_bbox)
            f_results = self._geofiltered_ordered_results(geofilter_bbox)
            return flask.jsonify({
                'results': f_results
            })

        @self.route('/_iqr_get_results_range')
        @self._parent_app.module_login.login_required
        def iqr_get_results_range():
            """
            Get a range of ordered results from the last refinement

            Optional geo-filtering bbox parameter: ``geofilter``
            - format: "...?[...]geofilter=x1,y1,x2,y2[&...]"
            - doesn't matter spatially where the two points are (ul, ll, ur, lr)

            If there hasn't been a refinement yet, this returns an empty list.

            We expect range bounds just like indexing a list: 0-base indexing,
            include/exclusive index bounds.

            Return list element format:
                (uid, probability)

            """
            s_idx = int(flask.request.args['s'])
            e_idx = int(flask.request.args['e'])
            geofilter_bbox = self._parse_geo_filter_arg(
                flask.request.args.get('geofilter', None)
            )
            self.log.info("Results fetch geo filter: %s", geofilter_bbox)
            f_results = self._geofiltered_ordered_results(geofilter_bbox, e_idx)
            return flask.jsonify({
                "results": f_results[s_idx:e_idx]
            })

        @self.route('/_iqr_get_random_id_order')
        @self._parent_app.module_login.login_required
        def iqr_get_random_id_order():
            """
            Return to the client a list of all known dataset IDs but in a random
            order. If there is currently an active IQR session with elements in
            its extension ingest, then those IDs are included in the random
            list.

            Optional geo-filtering bbox parameter: ``geofilter``
            - format: "...?[...]geofilter=x1,y1,x2,y2[&...]"
            - doesn't matter spatially where the two points are (ul, ll, ur, lr)

            """
            rand_ids = list(self._primary_ingest.ids())

            iqr_md = self.get_iqr_session_metadata()
            if iqr_md:
                with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                    rand_ids.extend(iqr_s.extension_ingest.ids())

            # randomize
            random.shuffle(rand_ids)

            return flask.jsonify({
                'num_ids': len(rand_ids),
                'rand_ids': rand_ids
            })

        @self.route('/_iqr_get_positive_id_groups')
        @self._parent_app.module_login.login_required
        def iqr_get_positive_id_groups():
            """
            Return a list of lists, where each inner list consists of item IDs
            that have been positively adjudicated and are sequential neighbors.

            Optional geo-filtering bbox parameter: ``geofilter``
            - format: "...?[...]geofilter=x1,y1,x2,y2[&...]"
            - doesn't matter spatially where the two points are (ul, ll, ur, lr)

            Return format:
            {
                "clusters": list of (<int:rank>, <int:id>)
            }

            """
            geofilter_bbox = self._parse_geo_filter_arg(
                flask.request.args.get('geofilter', None)
            )
            filtered_results = self._geofiltered_ordered_results(geofilter_bbox)

            # fetch positively adjudicated IDs from available IQR session
            pos_ids = set()
            iqr_md = self.get_iqr_session_metadata()
            if iqr_md:
                with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                    pos_ids = iqr_s.positive_ids

            pos_clusters = []
            cur_cluster = []
            idx = 0
            for uid, prob in filtered_results:
                if uid in pos_ids:
                    cur_cluster.append((idx, int(uid)))  # add UID
                elif cur_cluster:
                    # push the current cluster if there's anything in it
                    pos_clusters.append(cur_cluster)
                    cur_cluster = []
                idx += 1
            # if there were pos elements at the end of the results list somehow
            if cur_cluster:
                pos_clusters.append(cur_cluster)

            return flask.jsonify({
                'clusters': pos_clusters
            })

        @self.route('/_get_ingest_image_data', methods=['GET'])
        @self._parent_app.module_login.login_required
        def get_ingest_image_data():
            """
            Return the base64 data for an image given a valid ID

            ``long_side`` value is 0 when the long side of the image is on the
            x-axis or both sides are of equal length, and 1 when the y-axis is
            longer.

            ``is_explicit`` value is True if this image is marked as explicit in
            the ingest and False if not.

            """
            img_id = int(flask.request.args['id'])
            self.log.debug("Fetching image data for ID: (%s) %s"
                           % (type(img_id), img_id))
            success = True
            message = None
            data = None
            ext = None
            long_side = None  # 0 if x longer or equal, 1 if y longer

            # Since we could be asked for an image that is either in the primary
            # ingest or in the iqr session's extension ingest, cascade through
            # the two
            img_path = self._primary_ingest.get_img_path(img_id)
            is_explicit = False
            if img_path is None:
                # fall back to extension ingest
                iqr_md = self.get_iqr_session_metadata()
                with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                    img_path = iqr_s.extension_ingest.get_img_path(img_id)
                    if img_path is None:
                        success = False
                        message = "No such image ID %d" % img_id
                    else:
                        message = "Found image in IQR extension ingest"
                        is_explicit = iqr_s.extension_ingest.is_explicit(img_id)

            else:
                message = "Found image in primary ingest"
                is_explicit = self._primary_ingest.is_explicit(img_id)

            if success:
                # probably good to make sure that we just found a path to
                # something that exists.
                if not osp.exists(img_path):
                    success = False
                    message = "Found an image path, but the path was not " \
                              "valid (%s)" % img_path
                else:
                    #: :type: PIL.Image
                    img = PIL.Image.open(img_path)
                    long_side = int(img.size[1] > img.size[0])
                    # img = PIL.Image.open(img_path)
                    data = base64.encodestring(open(img_path, 'rb').read())
                    ext = osp.splitext(img_path)[1].lstrip('.')

            return flask.jsonify({
                # Boolean success marker
                "success": success,
                # Informational message
                "message": message,
                # Base-64 encoded image data
                "data": data,
                # Extension of the preview image
                "ext": ext,
                # Which side of the preview image is the longest, see above
                "long_side": long_side,
                # If this preview is of an explicit nature
                "is_explicit": is_explicit,
            })

        @self.route("/_get_ingest_image_metadata")
        @self._parent_app.module_login.login_required
        def get_ingest_image_metadata():
            """
            Return JSON metadata associated with a given ingest ID.

            ``message`` is only defined if success is False.

            """
            img_id = int(flask.request.args['id'])

            success = True
            message = None
            metadata = None

            # could be in primary or extension ingest
            img_md = self._primary_ingest.get_metadata(img_id)
            self.log.info("Primary ingest: %s", img_md)
            if img_md is None:
                iqr_md = self.get_iqr_session_metadata()
                if iqr_md:
                    with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                        img_md = iqr_s.extension_ingest.get_metadata(img_id)
                        self.log.info("Extension ingest: %s", img_md)
                        if img_md is None:
                            success = False
                            message = "No metadata for image ID %d" % img_id

            if success:
                metadata = img_md.as_json_dict()

            return flask.jsonify({
                "success": success,
                "message": message,
                "metadata": metadata,
            })

        @self.route("/_mark_img_explicit")
        @self._parent_app.module_login.login_required
        def mark_image_explicit():
            """
            Mark an ingest entry as explicit given an image id
            """
            img_id = int(flask.request.args['id'])

            if self._primary_ingest.has_id(img_id):
                self._primary_ingest.set_explicit(img_id)
            else:
                iqr_md = self.get_iqr_session_metadata()
                if img_id:
                    with self._iqr_controller.with_session(iqr_md.uuid) as iqrs:
                        iqrs.extension_ingest.set_explicit(img_id)

            return flask.jsonify({
                'success': True
            })

        @self.route('/_get_leaflet_tilelayer_info')
        @self._parent_app.module_login.login_required
        def get_leaflet_tilelayer_info():
            return flask.jsonify({
                "server": self._parent_app.config['LEAFLET_TILE_SERVER'],
                "attribution": self._parent_app.config['LEAFLET_ATTRIBUTION'],
                "maxZoom": self._parent_app.config['LEAFLET_MAX_ZOOM'],
            })

        @self.route('/_get_map_top_n')
        @self._parent_app.module_login.login_required
        def get_map_top_n():
            """
            Simply return the number of top results to show on the map.
            """
            return flask.jsonify({
                'n': int(self._parent_app.config['MAP_TOP_N'])
            })

    #
    # IQR Session metadata management
    #

    def get_iqr_session_metadata(self):
        """
        Get the metadata content from the current Flask secure cookie session

        :return: IQR Session cookie metadata, or None if there currently isn't
            any or the session is stale (referenced IQR session is not valid).
        :rtype: IqrSessionCookieMetadata or None

        """
        md = flask.session['user'].get(self.SESSION_KEY_IQR, None)
        if md:
            uid = md['uuid']
            if self._iqr_controller.has_session_uuid(uid):
                iqr_md = IqrSessionCookieMetadata(uid, md['type'])
                return iqr_md
            else:
                # session metadata is stale, remove it.
                self.remove_iqr_session_metadata()
                return None
        return None

    def set_iqr_session_metadata(self, metadata_obj):
        """
        Set new IQR session metadata.

        Should already be within with the iqr md write lock, but acquiring again
        for kicks (reentrant, so its not a problem)

        :param metadata_obj: IqrSessionCookieMetadata to set
        :type metadata_obj: IqrSessionCookieMetadata

        """
        self.remove_iqr_session_metadata()
        flask.session['user'][self.SESSION_KEY_IQR] = {
            'uuid': metadata_obj.uuid,
            'type': metadata_obj.type,
        }

    def remove_iqr_session_metadata(self):
        """
        Clear stored IQR Session metadata in the current secure cookie session
        """
        if self.SESSION_KEY_IQR in flask.session['user']:
            del flask.session['user'][self.SESSION_KEY_IQR]

    #
    # IQR Session manipulation
    #

    def iqr_session_init_new(self, descriptor_type_str=None):
        """
        Initialize a new IQR session and register it with the current session
        if there is no current IQR session registered.

        :raises RuntimeError: IQR Session already registered to the session
        :raises KeyError: Given descriptor type was not valid

        :param descriptor_type_str: Descriptor type label to create the
            IqrSession with.
        :type descriptor_type_str: str

        :return: UUID of the created IqrSession
        :rtype: uuid.UUID

        """
        # Current system is a state-based IQR Session based on secure cookie
        # Fail if there is currently an IQR Session set
        if self.get_iqr_session_metadata():
            raise RuntimeError("An existing IQR Session was found. Remove "
                               "first before initializing a new one.")
        else:
            # noinspection PyCallingNonCallable
            fd = self.FEATURE_DESCRIPTOR_TYPES[descriptor_type_str](
                self._parent_app.config['DIR_DATA'],
                self._parent_app.config['DIR_WORK']
            )
            uid = self._iqr_controller.init_new_session(fd)
            self.set_iqr_session_metadata(
                IqrSessionCookieMetadata(uid, descriptor_type_str))
            return uid

    def iqr_session_remove(self):
        """
        Remove the current IQR session and session record in the current secure
        cookie.

        :raises KeyError: There was no IqrSession registered in the session
            cookie.

        :return: True if a session was removed, false if nothing happened (no
            session to remove).
        :rtype: bool

        """
        iqr_md = self.get_iqr_session_metadata()
        if iqr_md:
            self._iqr_controller.remove_session(iqr_md.uuid)
            self.remove_iqr_session_metadata()
        return iqr_md is not None

    def iqr_session_ingest_files(self, *img_files):
        """
        Ingest one or more image files into the current IQR session.

        Already ingested files are ignored.

        :param img_files: Tuple of image file paths to ingest into the
            current IQR session
        :type img_files: tuple of str

        """
        # image file may already be ingested
        iqr_md = self.get_iqr_session_metadata()
        with self._iqr_controller.with_session(iqr_md.uuid) as iqr_session:
            iqr_session.extend_model(*img_files)

    #
    # Utility functions
    #

    # noinspection PyMethodMayBeStatic
    def _write_file_chunks(self, chunk_map, file_extension=''):
        """
        Given a mapping of chunks, write their contents to a temporary file,
        returning the path to that file.

        Returned file path should be manually removed by the user.

        :param chunk_map: Mapping of integer index to file-like chunk
        :type chunk_map: dict of (int, StringIO)
        :param file_extension: String extension to suffix the temporary file
            with
        :type file_extension: str

        :raises OSError: OS problems creating temporary file or writing it out.

        :return: Path to temporary combined file
        :rtype: str

        """
        tmp_fd, tmp_path = tempfile.mkstemp(file_extension)
        self.log.debug("Combining chunks into temporary file: %s", tmp_path)
        tmp_file = os.fdopen(tmp_fd, 'wb')
        for idx, chunk in sorted(chunk_map.items(), key=lambda p: p[0]):
            data = chunk.read()
            tmp_file.write(data)
        tmp_file.close()  # apparently also closes file descriptor?
        return tmp_path

    def _resolve_input_images(self, input_file_path):
        """
        When given a video, split it into frames and return a list of file paths
        to all saved frame (temporary location).

        NOTE: Currently only naively assume that if the extension is .mp4 its a
        video file. Assuming that everything else is an image file of some sort.

        :param input_file_path: Path to an image or video file
        :type input_file_path: str

        :return: Tuple of image frames to ingest
        :rtype: tuple of str

        """
        file_ext = osp.splitext(input_file_path)[1]

        if file_ext in ['.mp4']:
            # video file
            raise NotImplementedError("Video frame extraction not implemented "
                                      "yet")
        else:
            # image file, just return tuple of one element
            return input_file_path,

    # noinspection PyMethodMayBeStatic
    def _parse_geo_filter_arg(self, geofilter_str):
        """
        Parse out the given standard geo-filter string into a simple bbox

        Assumed string format: "x1,y1,x2,y2"

        If a bounding box could not be constructed from the given geofilter_str,
        None is returned

        :param geofilter_str: String providing 4 comma-separated floating point
            numbers.
        :type geofilter_str:

        :return: Simple bounding box representing the string provided
        :rtype: SimpleBoundingBox or None

        """
        # noinspection PyBroadException
        try:
            components = [float(e.strip()) for e in geofilter_str.split(',')]
            return SimpleBoundingBox(components[0:2], components[2:4])
        except:
            return None

    def _geofiltered_ordered_results(self, geofilter_bbox=None, limit=None):
        """
        Get current session's geo-filtered, ordered results.

        Returns empty list of there is no session or of there are currently no
        results.

        :param geofilter_bbox: Bounding box to filter on or None
        :type geofilter_bbox: SimpleBoundingBox or None
        :param limit: Limit the number of results we gather to improve
            performance
        :type limit: int or None

        :return: List of (uid, prob) filtered on the given bounding box. List
            will be in order of probability.
        :rtype: list of (int, float)

        """
        o_results = []

        iqr_md = self.get_iqr_session_metadata()
        if iqr_md:
            with self._iqr_controller.with_session(iqr_md.uuid) as iqr_s:
                o_results = iqr_s.ordered_results or []

        # filter
        filtered_results = []
        if geofilter_bbox:
            # TODO: Always add positively adjudicated items when encountered
            #       regardless of geo-position? Uploaded items will not have
            #       metadata, so they will never be kept when geo-filtering.

            # bind limit to either limit given or total number of results, which
            # ever is smaller
            limit = (limit and min(limit, len(o_results))) or len(o_results)

            # Assuming no metadata is coming along with extension files, so only
            # querying primary ingest
            i = 0  # results index runner
            while len(filtered_results) < limit and i < len(o_results):
                r_md = self._primary_ingest.get_metadata(o_results[i][0])
                if r_md and geofilter_bbox.contains_point(*r_md.get_geo_loc()):
                    filtered_results.append(o_results[i])

                i += 1

        else:
            filtered_results = o_results[:limit] if limit else o_results

        return filtered_results
