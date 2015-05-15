__author__ = 'purg'

import abc


class DataSet (object):
    """
    Base abstract class for data sets.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def count(self):
        """
        :return: The number of data elements in this set.
        :rtype: int
        """
        return

    @abc.abstractmethod
    def has_uuid(self, uuid):
        """
        Test if the given uuid refers to an element in this data set.

        :param uuid: Unique ID to test for inclusion. This should match the type
            that the set implementation expects or cares about.

        :return: True if the given uuid matches an element in this set, or False
            if it does not.
        :rtype: bool

        """
        return

    @abc.abstractmethod
    def get_data(self, uuid):
        """
        Get the data element the given uuid references, or raise an exception if
        the uuid does not reference any element in this set.

        :raises KeyError: If the given uuid does not refer to an element in this
            data set.

        :param uuid: The uuid of the element to retrieve.

        :return: The data element instance for the given uuid.
        :rtype: smqtk.data_rep.DataElement

        """
        return
