"""
Stub abstract class implementations.
"""
from smqtk.algorithms import Classifier, DescriptorGenerator, \
    NearestNeighborsIndex
from smqtk.representation import DescriptorIndex


STUB_MODULE_PATH = __name__


class StubDescriptorIndex (DescriptorIndex):
    @classmethod
    def is_usable(cls):
        return True

    def add_many_descriptors(self, descriptors):
        pass

    def count(self):
        pass

    def iteritems(self):
        pass

    def clear(self):
        pass

    def remove_descriptor(self, uuid):
        pass

    def remove_many_descriptors(self, uuids):
        pass

    def get_config(self):
        pass

    def get_many_descriptors(self, uuids):
        pass

    def has_descriptor(self, uuid):
        pass

    def iterkeys(self):
        pass

    def get_descriptor(self, uuid):
        pass

    def iterdescriptors(self):
        pass

    def add_descriptor(self, descriptor):
        pass


class StubClassifier (Classifier):
    """
    Classifier stub for testing IqrService.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def get_labels(self):
        pass

    def _classify(self, d):
        pass


class StubDescrGenerator (DescriptorGenerator):
    """
    DescriptorGenerator stub for testing IqrService.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def valid_content_types(self):
        pass

    def _compute_descriptor(self, data):
        pass


class StubNearestNeighborIndex (NearestNeighborsIndex):
    """
    NearestNeighborIndex stub for testing IqrService.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        pass

    def count(self):
        pass

    def _build_index(self, descriptors):
        pass

    def _update_index(self, descriptors):
        pass

    def _remove_from_index(self, uids):
        pass

    def _nn(self, d, n=1):
        pass
