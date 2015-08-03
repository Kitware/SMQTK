__author__ = 'purg'

import os
from smqtk.data_rep import DataElement
from smqtk.utils.plugin import get_plugins

def get_data_element_impls():
    """
    Discover and return DataElement implementation classes found in the plugin
    directory. Keys in the returned map are the names of the discovered classes
    and the paired values are the actual class type objects.

    We look for modules (directories or files) that start with and alphanumeric
    character ('_' prefixed files/directories are hidden, but not recommended).

    Within a module, we first look for a helper variable by the name
    ``DATA_ELEMENT_CLASS``, which can either be a single class object or an
    iterable of class objects, to be exported. If the variable is set to None,
    we skip that module and do not import anything. If the variable is not
    present, we look for a class by the same na e and casing as the module's
    name. If neither are found, the module is skipped.

    :return: Map of discovered class objects of type ``DataElement`` whose keys
        are the string names of the classes.
    :rtype: dict[str, type]

    """

    this_dir = os.path.abspath(os.path.dirname(__file__))
    helper_var = "DATA_ELEMENT_CLASS"
    return get_plugins(__name__, this_dir, helper_var, DataElement)
