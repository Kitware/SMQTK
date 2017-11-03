import os

from smqtk.utils.plugin import get_plugins
from ._interface_nn_index import NearestNeighborsIndex


__all__ = [
    'NearestNeighborsIndex', 'get_nn_index_impls',
]


def get_nn_index_impls(reload_modules=False):
    """
    Discover and return discovered ``NearestNeighborsIndex`` classes. Keys in
    the returned map are the names of the discovered classes, and the paired
    values are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable ``NN_INDEX_PATH``
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH
              separator character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``NN_INDEX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type ``NearestNeighborsIndex``
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "NN_INDEX_PATH"
    helper_var = "NN_INDEX_CLASS"
    return get_plugins(__name__, this_dir, env_var, helper_var,
                       NearestNeighborsIndex, reload_modules=reload_modules)
