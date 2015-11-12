import tempfile
import multiprocessing
import multiprocessing.pool
import os
import subprocess

import jinja2

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.utils.file_utils import iter_csv_file


__author__ = 'paul.tunison@kitware.com'


class CaffeDefaultImageNet (DescriptorGenerator):
    """
    Descriptor generator using the pre-trained AlexNet CNN network, yielding
    a 4096 length descriptor vector.

    Additional large files must be downloaded from Caffe in order to use this
    descriptor generator:
        - BVLC reference CaffeNet (233MB)
            http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
        - ImageNet image mean (17MB)
            extract the ``imagenet_mean.binaryproto`` file from:
                http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

    The paths to these files should be passed to the
    ``blvc_reference_caffenet_model_fp`` and ``image_mean_binary_fp`` constructor
    parameters respectively.

    This implementation is currently intended to be run on a GPU, though it may
    work on CPU-only caffe installations.

    """

    CNN_EXE = 'cnn_feature_extractor'

    PROTOTEXT_TEMPLATE = jinja2.Template(
        open(os.path.join(os.path.dirname(__file__),
                          "cnn_config.prototxt.tmpl")).read()
    )

    @classmethod
    def is_usable(cls):
        log = cls.logger()

        if not hasattr(CaffeDefaultImageNet, "_is_usable_cache"):
            CaffeDefaultImageNet._is_usable_cache = True

            try:
                subprocess.call([cls.CNN_EXE],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except OSError:
                log.warn("Could not location Caffe CNN descriptor generation "
                         "executable (\'%s\'). Make sure that SMQTK was build "
                         "successfully with the Caffe CNN executable option "
                         "turned ON." % cls.CNN_EXE)
                CaffeDefaultImageNet._is_usable_cache = False

        return CaffeDefaultImageNet._is_usable_cache

    def __init__(self, blvc_reference_caffenet_model, image_mean_binary,
                 gpu_batch_size,
                 layer_extraction='fc7',
                 temp_directory=None,
                 force_gpu=False,
                 cnn_exe=None):
        """
        :param blvc_reference_caffenet_model: Path to the BVLC model file.

        :param image_mean_binary: Path to the ImageNet image mean binary file.

        :param gpu_batch_size: Number of concurrent images to send to the GPU at
            a time. This is dependent on the RAM available to your GPU. If this
            is set too high, the executable may segfault due to an out-of-memory
            failure.

        :param layer_extraction: Layer of the CNN to extract as the feature
            vector.

        :param temp_directory: Optional directory to store temporary working
            files.

        :param force_gpu: Force CNN computation on the GPU. Executable must be
            built with this functionality enabled.

        :param cnn_exe: Custom name or path to the executable to use. When None
            we use the default executable name specified on the class in
            ``CNN_EXE``.

        """
        self.blvc_reference_caffenet_model_fp = blvc_reference_caffenet_model
        self.image_mean_binary_fp = image_mean_binary
        self.gpu_batch_size = gpu_batch_size
        self.layer_extraction = layer_extraction
        self.temp_directory = temp_directory
        self.force_gpu = force_gpu
        self.cnn_exe = cnn_exe or self.CNN_EXE

    def get_config(self):
        return {
            "blvc_reference_caffenet_model": self.blvc_reference_caffenet_model_fp,
            "image_mean_binary": self.image_mean_binary_fp,
            "gpu_batch_size": self.gpu_batch_size,
            "layer_extraction": self.layer_extraction,
            "temp_directory": self.temp_directory,
            "force_gpu": self.force_gpu,
            "cnn_exe": self.cnn_exe,
        }

    def valid_content_types(self):
        return {
            'image/tiff',
            'image/png',
            'image/jpeg',
        }

    def _compute_descriptor(self, data):
        raise NotImplementedError("Shouldn't get here as "
                                  "compute_descriptor[_async] is being "
                                  "overridden")

    def compute_descriptor(self, data, descr_factory, overwrite=False):
        """
        Given some kind of data, return a descriptor element containing a
        descriptor vector.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor.

        :param data: Some kind of input data for the feature descriptor.
        :type data: smqtk.representation.DataElement

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instance.
        :type descr_factory: smqtk.representation.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vector for the given data even when there exists a precomputed
            vector in the generated DescriptorElement as generated from the
            provided factory. This will overwrite the persistently stored vector
            if the provided factory produces a DescriptorElement implementation
            with such storage.
        :type overwrite: bool

        :return: Result descriptor element. UUID of this output descriptor is
            the same as the UUID of the input data element.
        :rtype: smqtk.representation.DescriptorElement

        """
        # TODO: Could write custom process for slight optimization
        #       over just calling the async version with one element.
        m = self.compute_descriptor_async([data], descr_factory, overwrite,
                                          procs=1)
        return m[data]

    def compute_descriptor_async(self, data_iter, descr_factory,
                                 overwrite=False, procs=None, **kwds):
        """
        Asynchronously compute feature data for multiple data items.

        This function does NOT use the class attribute PARALLEL for determining
        parallel factor as this method can take that specification as an
        argument.

        :param data_iter: Iterable of data elements to compute features for.
            These must have UIDs assigned for feature association in return
            value.
        :type data_iter: collections.Iterable[smqtk.representation.DataElement]

        :param descr_factory: Factory instance to produce the wrapping
            descriptor element instances.
        :type descr_factory: smqtk.representation.DescriptorElementFactory

        :param overwrite: Whether or not to force re-computation of a descriptor
            vectors for the given data even when there exists precomputed
            vectors in the generated DescriptorElements as generated from the
            provided factory. This will overwrite the persistently stored
            vectors if the provided factory produces a DescriptorElement
            implementation such storage.
        :type overwrite: bool

        :param procs: Optional specification of how many processors to use
            when pooling sub-tasks. If None, we attempt to use all available
            cores.
        :type procs: int

        :return: Mapping of input DataElement instances to the computed
            descriptor element.
            DescriptorElement UUID's are congruent with the UUID of the data
            element it is the descriptor of.
        :rtype: dict[smqtk.representation.DataElement,
                     smqtk.representation.DescriptorElement]

        """
        # Create DescriptorElement instances for each data elem.
        #: :type: dict[collections.Hashable, smqtk.representation.DataElement]
        data_elements = {}
        #: :type: dict[collections.Hashable, smqtk.representation.DescriptorElement]
        descr_elements = {}
        self._log.debug("Checking content types; aggregating data/descriptor "
                        "elements.")
        for d in data_iter:
            ct = d.content_type()
            if ct not in self.valid_content_types():
                raise ValueError("Cannot compute descriptor of content type "
                                 "'%s'" % ct)

            data_elements[d.uuid()] = d
            descr_elements[d.uuid()] = descr_factory.new_descriptor(self.name, d.uuid())

        # Reduce procs down to the number of elements to process if its smaller
        if len(data_elements) < (procs or multiprocessing.cpu_count()):
            procs = len(data_elements)

        # Queue up for processing UUID keys for data that don't have descriptors
        #   computed for it yet.
        #: :type: list[collections.Hashable]
        key_queue = []
        for d in descr_elements.itervalues():
            if overwrite or not d.has_vector():
                key_queue.append(d.uuid())

        if key_queue:
            # Split keys into two batches:
            #   - one evenly divisible by configured GPU batch size
            #   - remaining keys (tail)
            main_batch_size = self.gpu_batch_size * (len(key_queue) // self.gpu_batch_size)
            main_vectors = \
                self._compute_even_batch(key_queue[:main_batch_size],
                                         data_elements, descr_elements,
                                         self.gpu_batch_size, procs)
            # Assigning vectors to appropriate descriptors
            for k, v in zip(key_queue[:main_batch_size], main_vectors):
                descr_elements[k].set_vector(v)

            # Compute trailing keys if the total queue wasn't evenly divisible
            # by the GPU batch size.
            if main_batch_size != len(key_queue):
                tail_size = len(key_queue) - main_batch_size
                tail_vectors = self._compute_even_batch(key_queue[-tail_size:],
                                                        data_elements,
                                                        descr_elements,
                                                        tail_size, procs)
                main_vectors.extend(tail_vectors)

        return dict((data_elements[k], descr_elements[k])
                    for k in data_elements)

    def _compute_even_batch(self, keys, data_elements, descriptor_elements,
                            gpu_batch_size, procs):
        """
        Helper for computing a number of elements that is easily subdivided by
        the GPU batch size.
        """
        p = multiprocessing.pool.ThreadPool(procs)
        list_filepath = tempfile.mkstemp(".txt", self.name+'.',
                                         self.temp_directory)[1]
        prototxt_filepath = tempfile.mkstemp('.prototxt', self.name+'.',
                                             self.temp_directory)[1]

        try:
            self._log.debug("Writing temp files (threaded)")
            p.map(CaffeDefaultImageNet._async_write_temp,
                  ((data_elements[k], self.temp_directory)
                   for k in keys))

            if len(keys) % gpu_batch_size != 0:
                raise ValueError("GPU batch size does not evenly divide ")
            mini_batch_size = len(keys) // gpu_batch_size

            self._log.debug("make image list file")
            with open(list_filepath, 'w') as list_file:
                for k in keys:
                    list_file.write(
                        data_elements[k].write_temp(self.temp_directory) +
                        '\n'
                    )

            self._log.debug("Generate prototxt file")
            prototxt_str = self.PROTOTEXT_TEMPLATE.render(**{
                "image_mean_filepath": self.image_mean_binary_fp,
                "image_filelist_filepath": list_filepath,
                "batch_size": gpu_batch_size,
            })
            with open(prototxt_filepath, 'w') as f:
                f.write(prototxt_str)

            self._log.debug("Computing descriptors")
            output_filebase = tempfile.mkstemp(prefix=self.name+'.',
                                               dir=self.temp_directory)[1]
            os.remove(output_filebase)
            # The expected output CSV file path that will actually get
            # generated.
            output_csv = output_filebase + '.csv'
            call_args = [
                self.cnn_exe, self.blvc_reference_caffenet_model_fp,
                prototxt_filepath, self.layer_extraction, output_filebase,
                str(mini_batch_size), 'csv'
            ]
            if self.force_gpu:
                call_args.append("GPU")
            self._log.debug("CNN call args: %s", call_args)

            proc_cnn = subprocess.Popen(call_args)
            rc = proc_cnn.wait()
            if rc:
                raise RuntimeError("Failed CNN descriptor generation "
                                   "execution with return code: %d"
                                   % rc)
            if not os.path.isfile(output_csv):
                raise RuntimeError("Expected output CSV file not found, "
                                   "but return code was 0.\n"
                                   "Expected path:  %s" % output_csv)

            return list(iter_csv_file(output_csv))

        finally:
            # Clean up resources used
            p.close()
            p.join()

            if list_filepath:
                os.remove(list_filepath)
            if prototxt_filepath:
                os.remove(prototxt_filepath)

    @staticmethod
    def _async_write_temp(packet):
        """
        :type packet: (smqtk.representation.DataElement, str | None)
        """
        data_element, output_dir = packet
        data_element.write_temp(output_dir)
