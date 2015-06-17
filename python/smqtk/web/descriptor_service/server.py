# -*- coding: utf-8 -*-
"""
LICENCE
-------
Copyright 2015 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import flask
import logging
import mimetypes
import multiprocessing
import os

from smqtk.data_rep.data_element_impl.file_element import DataFileElement
from smqtk.data_rep.data_element_impl.memory_element import DataMemoryElement
from smqtk.data_rep.data_element_impl.url_element import DataUrlElement
from smqtk.utils import SimpleTimer
from smqtk.utils.configuration import (
    ContentDescriptorConfiguration,
    DescriptorFactoryConfiguration,
)


MIMETYPES = mimetypes.MimeTypes()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class DescriptorServiceServer (flask.Flask):
    """
    Simple server that takes in a specification of the following form:

        /<descriptor_type>/<uri>[?...]

    See the docstring for the ``compute_descriptor()`` method for complete rules
    on how to form a calling URL.

    Computes the requested descriptor for the given file and returns that via
    a JSON structure.

    Standard return JSON:
    {
        "success": <bool>,
        "descriptor": [ <float>, ... ]
        "message": <string>,
        "reference_uri": <uri>
    }

    # Additional Configuration

    ## DescriptorElementFactory type
    We will look for an environment variable `DSS_DE_FACTORY` for a string that
    corresponds to a label present in `DescriptorElementFactories` section
    of the `system_config.json` configuration file. If this is not found we will
    error requesting it's presence.

    """

    # Optional environment variable that can point to a configuration file
    ENV_CONFIG = "DescriptorService_CONFIG"

    # Environment variable to look at for descriptor element factory
    #   configuration
    ENV_DSS_DE_FACTORY = "DSS_DE_FACTORY"

    @property
    def log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    def __init__(self, config_filepath=None):
        super(DescriptorServiceServer, self).__init__(
            self.__class__.__name__,
            static_folder=os.path.join(SCRIPT_DIR, 'static'),
            template_folder=os.path.join(SCRIPT_DIR, 'templates')
        )

        #
        # Configuration setup
        #
        config_env_loaded = config_file_loaded = None

        # Load default -- This should always be present, aka base defaults
        self.config.from_object('smqtk_config')
        config_default_loaded = True

        # Load from env var if present
        if self.ENV_CONFIG in os.environ:
            self.log.info("Loading config from env var (%s)...",
                          self.ENV_CONFIG)
            self.config.from_envvar(self.ENV_CONFIG)
            config_env_loaded = True

        # Load from configuration file if given
        if config_filepath and os.path.isfile(config_filepath):
            config_file_path = \
                os.path.expanduser(os.path.abspath(config_filepath))
            self.log.info("Loading config from file (%s)...", config_file_path)
            self.config.from_pyfile(config_file_path)
            config_file_loaded = True

        self.log.debug("Config defaults loaded : %s", config_default_loaded)
        self.log.debug("Config from env loaded : %s", config_env_loaded)
        self.log.debug("Config from file loaded: %s", config_file_loaded)
        if not (config_default_loaded or config_env_loaded
                or config_file_loaded):
            raise RuntimeError("No configuration file specified for loading. "
                               "(%s=%s) (file=%s)"
                               % (self.ENV_CONFIG,
                                  os.environ.get(self.ENV_CONFIG, None),
                                  config_filepath))

        # Descriptor factory setup
        if self.ENV_DSS_DE_FACTORY not in os.environ:
            raise RuntimeError("Missing environment configuration variable "
                               "`%s`, which should be set to the configuration "
                               "label of the DescriptorElementFactory to use."
                               % self.ENV_DSS_DE_FACTORY)
        self.de_factory_label = os.environ.get(self.ENV_DSS_DE_FACTORY,
                                               "MemoryDescriptorFactory")
        self.log.info("Using Descriptor factory: \"%s\"", self.de_factory_label)
        try:
            self.descr_elem_factory = \
                DescriptorFactoryConfiguration.new_inst(self.de_factory_label)
        except KeyError:
            raise ValueError("Invalid factory label set to %s: \"%s\""
                             % (self.ENV_DSS_DE_FACTORY, self.de_factory_label))

        # Cache of ContentDescriptor instances
        self.descriptor_cache = {}
        self.descriptor_cache_lock = multiprocessing.RLock()

        #
        # Security
        #
        self.secret_key = self.config['SECRET_KEY']

        @self.route("/")
        def list_ingest_labels():
            return flask.jsonify({
                "labels": sorted(ContentDescriptorConfiguration
                                 .available_labels())
            })

        @self.route("/all/content_types")
        def all_content_types():
            """
            Of available descriptors, what content types are processable, and
            what types are associated to which available descriptor generator.
            """
            r = {}
            all_types = set()
            for l in ContentDescriptorConfiguration.available_labels():
                d = self.get_descriptor_inst(l)
                all_types.update(d.valid_content_types())
                r[l] = sorted(d.valid_content_types())

            return flask.jsonify({
                "all": sorted(all_types),
                "labels": r
            })

        @self.route("/all/compute/<path:uri>")
        def all_compute(uri):
            """
            Compute descriptors over the specified content for all generators
            that function over the data's content type.

            # JSON Return format
                {
                    "success": <bool>

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
                for l in ContentDescriptorConfiguration.available_labels():
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
                finished_loop = True

            return flask.jsonify({
                "success": finished_loop,
                "message": message,
                "descriptors": descriptors,
                "reference_uri": uri
            })

        @self.route("/<string:descriptor_label>/<path:uri>")
        def compute_descriptor(descriptor_label, uri):
            """
            # Data modes for upload/use
                - local filepath
                - base64
                - http/s URL

            The following sub-sections detail how different URI's can be used.

            ## Local Filepath
            The URI string must be prefixed with ``file://``, followed by the
            full path to the data file to describe.

            ## Base 64 data
            The URI string must be prefixed with "base64://", followed by the
            base64 encoded string. This mode also requires an additional
            ``?content_type=`` to provide data content type information. This
            mode saves the encoded data to temporary file for processing.

            ## HTTP/S address
            This is the default mode when the URI prefix is none of the above.
            This uses the requests module to locally download a data file
            for processing.

            # JSON Return format
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

    def get_descriptor_inst(self, label):
        """
        Get the cached content descriptor instance for a configuration label
        :type label: str
        :rtype: smqtk.content_description.ContentDescriptor
        """
        with self.descriptor_cache_lock:
            if label not in self.descriptor_cache:
                self.log.debug("Caching descriptor '%s'", label)
                self.descriptor_cache[label] = \
                    ContentDescriptorConfiguration.new_inst(label)

            return self.descriptor_cache[label]

    def resolve_data_element(self, uri):
        """
        Given the URI to some data, resolve it down to a DataElement instance.

        :raises ValueError: Issue with the given URI regarding either URI source
            resolution or data resolution.

        :param uri: URI to data
        :type uri: str
        :return: DataElement instance wrapping given URI to data.
        :rtype: smqtk.data_rep.DataElement

        """
        # Resolve URI into appropriate DataElement instance
        if uri[:7] == "file://":
            self.log.debug("Given local disk filepath")
            filepath = uri[7:]
            if not os.path.isfile(filepath):
                raise ValueError("File URI did not point to an existing file "
                                 "on disk.")
            else:
                de = DataFileElement(filepath)

        elif uri[:9] == "base64://":
            self.log.debug("Given base64 string")
            content_type = flask.request.args.get('content_type', None)
            self.log.debug("Content type: %s", content_type)
            if not content_type:
                raise ValueError("No content-type with given base64 data")
            else:
                b64str = uri[9:]
                de = DataMemoryElement.from_base64(b64str, content_type)

        else:
            self.log.debug("Given URL")
            de = DataUrlElement(uri)

        return de

    def generate_descriptor(self, de, cd_label):
        """
        Generate a descriptor for the content pointed to by the given URI using
        the specified descriptor generator.

        :raises ValueError: Content type mismatch given the descriptor generator
        :raises RuntimeError: Descriptor extraction failure.

        :type de: smqtk.data_rep.DataElement
        :type cd_label: str

        :return: Generated descriptor element instance with vector information.
        :rtype: smqtk.data_rep.DescriptorElement

        """
        with SimpleTimer("Computing descriptor...", self.log.debug):
            cd = self.get_descriptor_inst(cd_label)
            descriptor = cd.compute_descriptor(de, self.descr_elem_factory)

        return descriptor

    def run(self, host=None, port=None, debug=False, **options):
        """
        Override of the run method, drawing running host and port from
        configuration by default. 'host' and 'port' values specified as argument
        or keyword will override the app configuration.
        """
        super(DescriptorServiceServer, self)\
            .run(host=(host or self.config['RUN_HOST']),
                 port=(port or self.config['RUN_PORT']),
                 **options)
