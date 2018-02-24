
import flask
import multiprocessing
import os
from six.moves import StringIO
import tempfile

from smqtk.utils import SmqtkObject
from smqtk.utils import file_utils


script_dir = os.path.dirname(os.path.abspath(__file__))


class FileUploadMod (SmqtkObject, flask.Blueprint):
    """
    Flask blueprint for file uploading.
    """

    def __init__(self, name, parent_app, working_directory, url_prefix=None):
        """
        Initialize uploading module

        :param parent_app: Parent Flask app
        :type parent_app: smqtk.Web.search_app.base_app.search_app

        :param working_directory: Directory for temporary file storage during
            upload up to the time a user takes control of the file.
        :type working_directory: str

        """
        super(FileUploadMod, self).__init__(
            name, __name__,
            static_folder=os.path.join(script_dir, 'static'),
            url_prefix=url_prefix
        )
        # TODO: Thread safety

        self.parent_app = parent_app
        self.working_dir = working_directory

        # TODO: Move chunk storage to database for APACHE multiprocessing
        # File chunk aggregation
        #   Top level key is the file ID of the upload. The dictionary
        #   underneath that is the index ID'd chunks. When all chunks are
        #   present, the file is written and the entry in this map is removed.
        #: :type: dict of (str, dict of (int, StringIO))
        self._file_chunks = {}
        # Lock per file ID so as to not collide when uploading multiple chunks
        #: :type: dict of (str, RLock)
        self._fid_locks = {}

        # FileID to temporary path that a completed file is located at.
        self._completed_files = {}

        #
        # Routing
        #

        @self.route('/upload_chunk', methods=["POST"])
        @self.parent_app.module_login.login_required
        def upload_file():
            """
            Handle arbitrary file upload to OS temporary file storage, recording
            file upload completions.

            """
            form = flask.request.form
            self._log.debug("POST form contents: %s" % str(flask.request.form))

            fid = form['flowIdentifier']
            success = True
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
                message = "Uploaded chunk #%d of %d for file '%s'" \
                    % (current_chunk, total_chunks, filename)

                if total_chunks == len(self._file_chunks[fid]):
                    self._log.debug("[%s] Final chunk uploaded",
                                   filename+"::"+fid)
                    # have all chucks in memory now
                    try:
                        # Combine chunks into single file
                        file_ext = os.path.splitext(filename)[1]
                        file_saved_path = self._write_file_chunks(
                            self._file_chunks[fid], file_ext
                        )
                        self._log.debug("[%s] saved from chunks: %s",
                                       filename+"::"+fid, file_saved_path)
                        # now in file, free up dict memory

                        self._completed_files[fid] = file_saved_path
                        message = "[%s] Completed upload" % (filename+"::"+fid)

                    except IOError as ex:
                        self._log.debug("[%s] Failed to write combined chunks",
                                       filename+"::"+fid)
                        success = False
                        message = "Failed to write out combined chunks for " \
                                  "file %s: %s" % (filename, str(ex))
                        raise RuntimeError(message)

                    except NotImplementedError as ex:
                        success = False
                        message = "Encountered non-implemented code path: %s" \
                                  % str(ex)
                        raise RuntimeError(message)

                    finally:
                        # remove chunk map entries
                        del self._file_chunks[fid]
                        del self._fid_locks[fid]

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

        @self.route("/completed_uploads")
        @self.parent_app.module_login.login_required
        def completed_uploads():
            return flask.jsonify(self._completed_files)

    def upload_post_url(self):
        """
        :return: The url string to give to the JS upload zone for POSTing file
            chunks.
        :rtype: str
        """
        return self.url_prefix + '/upload_chunk'

    def get_path_for_id(self, file_unique_id):
        """
        Get the path to the temp file that was uploaded.

        It is the user's responsibility to remove this file when it is done
        being used, or move it else where.

        :param file_unique_id: Unique ID of the uploaded file
        :type file_unique_id: str

        :return: The path to the complete uploaded file.

        """
        return self._completed_files[file_unique_id]

    def clear_completed(self, file_unique_id):
        """
        Clear the completed file entry in our cache. This should be called after
        taking responsibility for an uploaded file.

        This does NOT delete the file.

        :raises KeyError: If the given unique ID does not correspond to an
            entry in our completed cache.

        :param file_unique_id: Unique ID of an uploaded file to clear from the
            completed cache.
        :type file_unique_id: str

        """
        del self._completed_files[file_unique_id]

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
        # Make sure write dir exists...
        if not os.path.isdir(self.working_dir):
            file_utils.safe_create_dir(self.working_dir)
        tmp_fd, tmp_path = tempfile.mkstemp(file_extension,
                                            dir=self.working_dir)
        self._log.debug("Combining chunks into temporary file: %s", tmp_path)
        tmp_file = open(tmp_path, 'wb')
        for idx, chunk in sorted(chunk_map.items(), key=lambda p: p[0]):
            data = chunk.read()
            tmp_file.write(data)
        tmp_file.close()
        return tmp_path
