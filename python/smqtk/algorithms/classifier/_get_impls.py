import os

from smqtk.utils import plugin

from ._interface_classifier import Classifier


def get_classifier_impls(reload_modules=False, sub_interface=None):
    """
    Discover and return discovered ``Classifier`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable
          :envvar:`CLASSIFIER_PATH`
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``CLASSIFIER_CLASS``, which can either be a single class object or an
    iterable of class objects, to be specifically exported. If the variable is
    set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :param sub_interface: Only return implementations that also descend from
        the given sub-interface. The given interface must also descend from
        :class:`Classifier`.

    :return: Map of discovered class object of type :class:`Classifier`
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "CLASSIFIER_PATH"
    helper_var = "CLASSIFIER_CLASS"
    if sub_interface is None:
        base_class = Classifier
    else:
        assert issubclass(sub_interface, Classifier), \
            "The given sub-interface type must descend from `Classifier`."
        base_class = sub_interface
    # __package__ resolves to the containing module of this module, or
    # `smqtk.algorithms.classifier` in this case.
    return plugin.get_plugins(__package__, this_dir, env_var, helper_var,
                              base_class, reload_modules=reload_modules)
