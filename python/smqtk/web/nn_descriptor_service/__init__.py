import mimetypes
import multiprocessing
import os

import flask
import requests

from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.algorithms.nn_index import NearestNeighborsIndex, get_nn_index_impls
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.utils import SimpleTimer
from smqtk.utils import plugin
from smqtk.utils.configuration import merge_configs
from smqtk.web import SmqtkWebApp

MIMETYPES = mimetypes.MimeTypes()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class NearestNeighborServiceServer (SmqtkWebApp):
    """
    Simple server that takes in a specification of the following form:

        /nn/<path:uri>[?...]

    Computes the nearest neighbor index for the given data and returns a list
    of nearest neighbors in the following format

    Standard return JSON:
    {
        "success": <bool>,
        "neighbors": [ <float>, ... ]
        "message": <string>,
        "reference_uri": <uri>
    }

    # Additional Configuration

    ## Environment variable hook
    We will look for an environment variable `DescriptorService_CONFIG` for a
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
        c = super(NearestNeighborServiceServer, cls).get_default_config()
        merge_configs(c, {
            "descriptor_factory": DescriptorElementFactory.get_default_config(),
            "descriptor_generators": {
                "example": plugin.make_config(get_descriptor_generator_impls)
            },
            "nn_index": plugin.make_config(get_nn_index_impls),
        })
        return c

    def __init__(self, json_config):
        """
        Initialize application based of supplied JSON configuration

        :param json_config: JSON configuration dictionary
        :type json_config: dict

        """
        super(NearestNeighborServiceServer, self).__init__(json_config)

        # Descriptor factory setup
        self.log.info("Initializing DescriptorElementFactory")
        self.descr_elem_factory = DescriptorElementFactory.from_config(
            self.json_config['descriptor_factory']
        )

        # Descriptor generator configuration labels
        #: :type: dict[str, dict]
        self.generator_config = self.json_config['descriptor_generator']

        self.nn_index = plugin.from_plugin_config(
            json_config['nn_index'],
            get_nn_index_impls
        )

        self.descriptor_generator_inst = plugin.from_plugin_config(
                                            self.generator_config,
                                            get_descriptor_generator_impls)


        @self.route("/nn/<path:uri>")
        def compute_nearest_neighbors(uri):
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

                    "neighbors": <None|list[float]>

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
                    descriptor = self.descriptor_generator_inst.\
                        compute_descriptor(de, self.descr_elem_factory)

                    if descriptor is not None:
                        descriptor.set_vector(descriptor.vector().flatten())
                    else
                        self.log.error("Descriptor is null or invalid")
                except RuntimeError, ex:
                    message = "Descriptor extraction failure: %s" % str(ex)
                except ValueError, ex:
                    message = "Data content type issue: %s" % str(ex)

            # fail here if de is None
            # Default is 8
            num_neighbors = flask.request.args.get("num_neighbors", 8)

            neighbors = []
            if descriptor is not None:
                neighbors, _ = self.nn_index.nn(descriptor, n=num_neighbors)

            # TODO: Return the optional descriptor vector for the neighbors
            return flask.jsonify({
                "success": descriptor is not None,
                "message": message,
                "neighbors": [n.uuid() for n in neighbors],
                "reference_uri": uri
            })

    def get_config(self):
        return self.json_config

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
            try:
                de = DataUrlElement(uri)
            except requests.HTTPError, ex:
                raise ValueError("Failed to initialize URL element due to "
                                 "HTTPError: %s" % str(ex))

        return de

APPLICATION_CLASS = NearestNeighborServiceServer
