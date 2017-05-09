import mimetypes
import os

import flask
import requests

from smqtk.algorithms import (
    get_descriptor_generator_impls,
    get_nn_index_impls
)
from smqtk.representation import (
    DescriptorElementFactory,
    get_descriptor_index_impls,
)
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element.url_element import DataUrlElement
from smqtk.utils import plugin
from smqtk.utils import merge_dict
from smqtk.web import SmqtkWebApp

MIMETYPES = mimetypes.MimeTypes()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class NearestNeighborServiceServer (SmqtkWebApp):
    """
    Simple server that takes in a specification of the following form:

        /nn/<path:uri>[?...]

    Computes the nearest neighbor index for the given data and returns a list
    of nearest neighbors in the following format

    Standard return JSON::
    {
        "success": <bool>,
        "neighbors": [ <float>, ... ]
        "message": <string>,
        "reference_uri": <uri>
    }
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
        merge_dict(c, {
            "descriptor_factory": DescriptorElementFactory.get_default_config(),
            "descriptor_generator":
                plugin.make_config(get_descriptor_generator_impls()),
            "nn_index": plugin.make_config(get_nn_index_impls()),
            "descriptor_index":
                plugin.make_config(get_descriptor_index_impls()),
            "update_descriptor_index": False,
        })
        return c

    def __init__(self, json_config):
        """
        Initialize application based of supplied JSON configuration

        :param json_config: JSON configuration dictionary
        :type json_config: dict

        """
        super(NearestNeighborServiceServer, self).__init__(json_config)

        self.update_index = json_config['update_descriptor_index']

        # Descriptor factory setup
        self._log.info("Initializing DescriptorElementFactory")
        self.descr_elem_factory = DescriptorElementFactory.from_config(
            self.json_config['descriptor_factory']
        )

        #: :type: smqtk.representation.DescriptorIndex | None
        self.descr_index = None
        if self.update_index:
            self._log.info("Initializing DescriptorIndex to update")
            #: :type: smqtk.representation.DescriptorIndex | None
            self.descr_index = plugin.from_plugin_config(
                json_config['descriptor_index'],
                get_descriptor_index_impls()
            )

        #: :type: smqtk.algorithms.NearestNeighborsIndex
        self.nn_index = plugin.from_plugin_config(
            json_config['nn_index'],
            get_nn_index_impls()
        )

        #: :type: smqtk.algorithms.DescriptorGenerator
        self.descriptor_generator_inst = plugin.from_plugin_config(
            self.json_config['descriptor_generator'],
            get_descriptor_generator_impls()
        )

        @self.route("/count", methods=['GET'])
        def count():
            """
            Return the number of elements represented in this index.
            """
            return flask.jsonify(**{
                "count": self.nn_index.count(),
            })

        @self.route("/compute/<path:uri>", methods=["POST"])
        def compute(uri):
            """
            Compute the descriptor for a URI specified data element using the
            configured descriptor generator.

            See ``compute_nearest_neighbors`` method docstring for URI
            specifications accepted.

            If the a descriptor index was configured and update was turned on,
            we add the computed descriptor to the index.

            JSON Return format::
                {
                    "success": <bool>

                    "message": <str>

                    "descriptor": <None|list[float]>

                    "reference_uri": <str>
                }

            :param uri: URI data specification.

            """
            descriptor = None
            try:
                descriptor = self.generate_descriptor_for_uri(uri)
                message = "Descriptor generated"
                descriptor = map(float, descriptor.vector())
            except ValueError as ex:
                message = "Input value issue: %s" % str(ex)
            except RuntimeError as ex:
                message = "Descriptor extraction failure: %s" % str(ex)

            return flask.jsonify(
                success=descriptor is not None,
                message=message,
                descriptor=descriptor,
                reference_uri=uri,
            )

        @self.route("/nn/<path:uri>")
        @self.route("/nn/n=<int:n>/<path:uri>")
        @self.route("/nn/n=<int:n>/<int:start_i>:<int:end_i>/<path:uri>")
        def compute_nearest_neighbors(uri, n=10, start_i=None, end_i=None):
            """
            Data modes for upload/use:

                - local filepath
                - base64
                - http/s URL
                - existing data/descriptor UUID

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

            Existing Data/Descriptor by UUID
            --------------------------------
            When given a uri prefixed with "uuid://", we interpret the remainder
            of the uri as the UUID of a descriptor already present in the
            configured descriptor index. If the given UUID is not present in the
            index, a KeyError is raised.

            JSON Return format
            ------------------
                {
                    "success": <bool>

                    "message": <str>

                    "neighbors": <None|list[float]>

                    "reference_uri": <str>
                }

            :param n: Number of neighbors to query for
            :param start_i: The starting index of the neighbor vectors to slice
                into for return.
            :param end_i: The ending index of the neighbor vectors to slice
                into for return.
            :type uri: str

            """
            descriptor = None
            try:
                descriptor = self.generate_descriptor_for_uri(uri)
                message = "descriptor computed"
            except ValueError as ex:
                message = "Input data issue: %s" % str(ex)
            except RuntimeError as ex:
                message = "Descriptor generation failure: %s" % str(ex)

            # Base pagination slicing based on provided start and end indices,
            # otherwise clamp to beginning/ending of queried neighbor sequence.
            page_slice = slice(start_i or 0, end_i or n)
            neighbors = []
            dists = []
            if descriptor is not None:
                try:
                    neighbors, dists = \
                        self.nn_index.nn(descriptor, n)
                except ValueError as ex:
                    message = "Descriptor or index related issue: %s" % str(ex)

            # TODO: Return the optional descriptor vectors for the neighbors
            # noinspection PyTypeChecker
            d = {
                "success": bool(descriptor is not None),
                "message": message,
                "neighbors": [n.uuid() for n in neighbors[page_slice]],
                "distances": dists[page_slice],
                "reference_uri": uri
            }
            return flask.jsonify(d)

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
        self._log.debug("Resolving URI: %s", uri)
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
            except requests.HTTPError as ex:
                raise ValueError("Failed to initialize URL element due to "
                                 "HTTPError: %s" % str(ex))

        return de

    def generate_descriptor_for_uri(self, uri):
        """
        Given the URI to some data, resolve it and compute its descriptor,
        returning a DescriptorElement.

        :param uri: URI to data
        :type uri: str

        :return: DescriptorElement instance of the generate descriptor.
        :rtype: smqtk.representation.DescriptorElement

        """
        # Short-cut if we are given data/descriptor UUID via URI
        if uri[:7] == 'uuid://':
            descriptor = self.descr_index[uri[7:]]
        else:
            de = self.resolve_data_element(uri)
            descriptor = self.descriptor_generator_inst.compute_descriptor(
                de, self.descr_elem_factory
            )
            if self.update_index:
                self._log.info("Updating index with new descriptor")
                self.descr_index.add_descriptor(descriptor)
            if not descriptor.has_vector():
                raise RuntimeError("No descriptor content")
        return descriptor


APPLICATION_CLASS = NearestNeighborServiceServer
