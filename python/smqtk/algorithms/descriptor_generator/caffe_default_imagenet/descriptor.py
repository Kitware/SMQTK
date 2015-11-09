from smqtk.algorithms.descriptor_generator import DescriptorGenerator


__author__ = 'paul.tunison@kitware.com'


class CaffeDefaultImageNet (DescriptorGenerator):
    """
    Descriptor generator using the pre-trained AlexNet CNN network, yielding
    a 4096 length descriptor vector.
    """

    EXE_PATH = "cnn_feature_extractor"

    @classmethod
    def is_usable(cls):
        # Try to call the executable which needs to be on the PATH
        return True

    def get_config(self):
        return {}

    def valid_content_types(self):
        return {
            'image/tiff',
            'image/png',
            'image/jpeg',
        }

    def _compute_descriptor(self, data):
        # For actual implementation, see the
        # bin/scripts/runners/memex_hbase_gpu_machine/compute.py file.
        raise NotImplementedError("Currently expecting only data that has been "
                                  "computed before")

    def compute_descriptor_async(self, data_iter, descr_factory,
                                 overwrite=False, **kwds):
        # Create DescriptorElement instances for each data elem.
        # Queue up for processing data that doesn't have descriptors computed
        #   for it yet.
        # Generate necessary files for exe, write temp files for data, compute
        #   batch descriptors (with GPU if a flag is set, constructor param).
        # Convert result types are return merged set.
        raise NotImplementedError("%s async descriptor computation will have a "
                                  "custom implementation, but this is not yet "
                                  "complete." % self.name)
