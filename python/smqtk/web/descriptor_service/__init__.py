import mimetypes
import multiprocessing
import os

import flask
import requests

from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.utils import SimpleTimer
from smqtk.utils import plugin
from smqtk.utils import merge_dict
from smqtk.web import SmqtkWebApp

MIMETYPES = mimetypes.MimeTypes()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class DescriptorServiceServer (SmqtkWebApp):
    """
    Simple server that takes in a specification of the following form:

        /<descriptor_type>/<uri>[?...]

    See the docstring for the ``compute_descriptor()`` method for complete rules
    on how to form a calling URL.

    Computes the requested descriptor for the given file and returns that via
    a JSON structure.

    Standard return JSON::

        {
            "success": <bool>,
            "descriptor": [ <float>, ... ]
            "message": <string>,
            "reference_uri": <uri>
        }

    Additional Configuration



    .. note:: We will look for an environment variable `DescriptorService_CONFIG` for a
              string file path to an additional JSON configuration file to consider.

    """

    @classmethod
    def is_usable(cls):
      return True

    @classmethod
    def get_default_config(cls):
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        :return: Default configuration dictionary for the class.
        :rtype: dict

        """
        c = super(DescriptorServiceServer, cls).get_default_config()
        merge_dict(c, {
            "descriptor_factory": DescriptorElementFactory.get_default_config(),
            "descriptor_generators": {
                "example": plugin.make_config(get_descriptor_generator_impls())
            }
        })
        return c

    def __init__(self, json_config):
        """
        Initialize application based of supplied JSON configuration

        :param json_config: JSON configuration dictionary
        :type json_config: dict

        """
        super(DescriptorServiceServer, self).__init__(json_config)

        # Descriptor factory setup
        self._log.info("Initializing DescriptorElementFactory")
        self.descr_elem_factory = DescriptorElementFactory.from_config(
            self.json_config['descriptor_factory']
        )

        # Descriptor generator configuration labels
        #: :type: dict[str, dict]
        self.generator_label_configs = self.json_config['descriptor_generators']

        # Cache of DescriptorGenerator instances so we don't have to continuously
        # initialize them as we get requests.
        self.descriptor_cache = {}
        self.descriptor_cache_lock = multiprocessing.RLock()

        @self.route("/")
        def list_ingest_labels():
            return flask.jsonify({
                "labels": sorted(self.generator_label_configs.iterkeys())
            })

        @self.route("/all/content_types")
        def all_content_types():
            """
            Of available descriptors, what content types are processable, and
            what types are associated to which available descriptor generator.
            """
            all_types = set()
            # Mapping of configuration label to content types that generator
            # can handle
            r = {}
            for l in self.generator_label_configs:
                d = self.get_descriptor_inst(l)
                all_types.update(d.valid_content_types())
                r[l] = sorted(d.valid_content_types())

            return flask.jsonify({
                "all": sorted(all_types),
                "by-label": r
            })

        @self.route("/all/compute/<path:uri>")
        def all_compute(uri):
            """
            Compute descriptors over the specified content for all generators
            that function over the data's content type.

            JSON Return format::

                {
                    "success": <bool>

                    "content_type": <str>

                    "message": <str>

                    "descriptors": {  "<label>":  <list[float]>, ... } | None

                    "reference_uri": <str>
                }

            """
            message = "execution nominal"

            data_elem = None
            try:
                data_elem = self.resolve_data_element(uri)
            except ValueError, ex:
                message = "Failed URI resolution: %s" % str(ex)

            descriptors = {}
            finished_loop = False
            if data_elem:
                for l in self.generator_label_configs:
                    if data_elem.content_type() \
                            in self.get_descriptor_inst(l).valid_content_types():
                        d = None
                        try:
                            d = self.generate_descriptor(data_elem, l)
                        except RuntimeError, ex:
                            message = "Descriptor extraction failure: %s" \
                                      % str(ex)
                        except ValueError, ex:
                            message = "Data content type issue: %s" % str(ex)

                        descriptors[l] = d and d.vector().tolist()
                if not descriptors:
                    message = "No descriptors can handle URI content type: %s" \
                              % data_elem.content_type
                else:
                    finished_loop = True

            return flask.jsonify({
                "success": finished_loop,
                "content_type": data_elem.content_type(),
                "message": message,
                "descriptors": descriptors,
                "reference_uri": uri
            })

        @self.route("/<string:descriptor_label>/<path:uri>")
        def compute_descriptor(descriptor_label, uri):
            """

            Data modes for upload/use::

                - local filepath
                - base64
                - http/s URL

            The following sub-sections detail how different URI's can be used.

            Local Filepath
            --------------

            The URI string must be prefixed with ``file://``, followed by the
            full path to the data file to describe.

            Base 64 data
            ------------

            The URI string must be prefixed with "base64://", followed by the
            base64 encoded string. This mode also requires an additional
            ``?content_type=`` to provide data content type information. This
            mode saves the encoded data to temporary file for processing.

            HTTP/S address
            --------------

            This is the default mode when the URI prefix is none of the above.
            This uses the requests module to locally download a data file
            for processing.

            JSON Return format::

                {
                    "success": <bool>

                    "message": <str>

                    "descriptor": <None|list[float]>

                    "reference_uri": <str>
                }

            :type descriptor_label: str
            :type uri: str

            """
            message = "execution nominal"
            descriptor = None

            de = None
            try:
                de = self.resolve_data_element(uri)
            except ValueError, ex:
                message = "URI resolution issue: %s" % str(ex)

            if de:
                try:
                    descriptor = self.generate_descriptor(de, descriptor_label)
                except RuntimeError, ex:
                    message = "Descriptor extraction failure: %s" % str(ex)
                except ValueError, ex:
                    message = "Data content type issue: %s" % str(ex)

            return flask.jsonify({
                "success": descriptor is not None,
                "message": message,
                "descriptor":
                    (descriptor is not None and descriptor.vector().tolist())
                    or None,
                "reference_uri": uri
            })

    def get_config(self):
        return self.json_config

    def get_descriptor_inst(self, label):
        """
        Get the cached content descriptor instance for a configuration label
        :type label: str
        :rtype: smqtk.descriptor_generator.DescriptorGenerator
        """
        with self.descriptor_cache_lock:
            if label not in self.descriptor_cache:
                self._log.debug("Caching descriptor '%s'", label)
                self.descriptor_cache[label] = \
                    plugin.from_plugin_config(
                    self.generator_label_configs[label],
                        get_descriptor_generator_impls()
                    )

            return self.descriptor_cache[label]

    def resolve_data_element(self, uri):
        """
        Given the URI to some data, resolve it down to a DataElement instance.

        :raises ValueError: Issue with the given URI regarding either URI source
            resolution or data resolution.

        :param uri: URI to data
        :type uri: str
        :return: DataElement instance wrapping given URI to data.
        :rtype: smqtk.representation.DataElement

        """
        # Resolve URI into appropriate DataElement instance
        if uri[:7] == "file://":
            self._log.debug("Given local disk filepath")
            filepath = uri[7:]
            if not os.path.isfile(filepath):
                raise ValueError("File URI did not point to an existing file "
                                 "on disk.")
            else:
                de = DataFileElement(filepath)

        elif uri[:9] == "base64://":
            self._log.debug("Given base64 string")
            content_type = flask.request.args.get('content_type', None)
            self._log.debug("Content type: %s", content_type)
            if not content_type:
                raise ValueError("No content-type with given base64 data")
            else:
                b64str = uri[9:]
                de = DataMemoryElement.from_base64(b64str, content_type)

        else:
            self._log.debug("Given URL")
            try:
                de = DataUrlElement(uri)
            except requests.HTTPError, ex:
                raise ValueError("Failed to initialize URL element due to "
                                 "HTTPError: %s" % str(ex))

        return de

    def generate_descriptor(self, de, cd_label):
        """
        Generate a descriptor for the content pointed to by the given URI using
        the specified descriptor generator.

        :raises ValueError: Content type mismatch given the descriptor generator
        :raises RuntimeError: Descriptor extraction failure.

        :type de: smqtk.representation.DataElement
        :type cd_label: str

        :return: Generated descriptor element instance with vector information.
        :rtype: smqtk.representation.DescriptorElement

        """
        with SimpleTimer("Computing descriptor...", self._log.debug):
            cd = self.get_descriptor_inst(cd_label)
            descriptor = cd.compute_descriptor(de, self.descr_elem_factory)

        return descriptor


APPLICATION_CLASS = DescriptorServiceServer
