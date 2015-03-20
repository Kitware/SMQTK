"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""


def get_classifiers():
    """
    Introspect this module and return a dictionary-like structure granting
    access to the various implementations in a logical, structured way.

    :return: mapping of classifier type to learner/searcher tagged classes in
        the form:
            { <module_name>: { 'learner': <class>, 'searcher': <class> },

              'svm': { 'learner': <class>, 'searcher': <class> },

              ... (etc.)
            }
    :rtype: dict of (str, dict of (str, type))

    """
    import logging
    import os
    import os.path as osp
    import re

    log = logging.getLogger(__name__)
    module_re = re.compile("^[a-zA-Z]\w+$")
    classifier_map = {}

    plugin_dir = osp.dirname(__file__)
    log.debug("all things: %s", os.listdir(plugin_dir))
    for classifier_name in os.listdir(plugin_dir):
        log.debug("trying: %s", classifier_name)
        if module_re.match(classifier_name):
            log.debug("matched: %s", classifier_name)
            ###
            # Import the element as a module (should be a directory)

            # construct full module name given the current module path
            module_name = '.'.join([__name__, classifier_name])
            try:
                module = __import__(module_name, fromlist=[__name__])
            except ImportError, ex:
                log.warning("Couldn't import module '%s'. Skipping. "
                            "(error: %s)",
                            module_name, str(ex))
                continue

            ###
            # Attempt to extract ``learner`` and ``searcher`` classes from
            # module.
            learner = getattr(module, 'learn', None)
            searcher = getattr(module, 'search', None)

            if None in (learner, searcher):
                log.warning("Classifier module '%s' incomplete in defining "
                            "learner and searcher classes. Skipping module",
                            classifier_name)
                continue

            classifier_map[classifier_name] = {}
            classifier_map[classifier_name]['learn'] = learner
            classifier_map[classifier_name]['search'] = searcher

    return classifier_map
