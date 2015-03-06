"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

from .. import VCDWorkerInterface


def get_descriptor_modules():
    """
    Discover and return modules found in this plug-in directory. Keys will be
    the name of the discovered module and the values will be the module's
    VCDWorkerInterface implementation class. Modules that will be discovered are
    python files that start with an alphanumeric character. Support
    functionality and/or files for a descriptor module may be hidden within a
    python file that starts with an underscore ('_') or in a sub-module.

    Each descriptor module file must define a variable "WORKER_CLASS" that is
    assigned the class of the VCDWorkerInterface implementation. If this is not
    found, we will assume that the name of the file is the name of the class,
    and we will look for that. If that is not found, and exception is thrown as
    this would be an invalid module definition.

    If the "WORKER_CLASS" variable has been explicitly set to None, the
    descriptor module will be ignored.

    :return: Map of discovered descriptor modules.
    :rtype: dict of (str, types.ModuleType)

    """
    import logging
    import re
    import os
    import os.path as osp

    log = logging.getLogger(__name__)
    module_map = {}

    plugin_dir = osp.dirname(__file__)
    file_re = re.compile("^[a-zA-Z].*\.py$")

    for _file in os.listdir(plugin_dir):
        if file_re.match(_file):
            log.debug("Matched file: %s", _file)
            file_basename = osp.splitext(_file)[0]

            # Importing module
            module_name = '.'.join((__name__, file_basename))
            try:
                module = __import__(module_name, fromlist=__name__)
            except ImportError, ex:
                import traceback
                log.warning("Couldn't import '%s' as a python module. "
                            "Skipping file. See traceback. (error: %s)\n"
                            "%s",
                            _file, str(ex), traceback.format_exc())
                continue

            # Look for WORKER_CLASS variable
            worker_class = None
            if hasattr(module, "WORKER_CLASS"):
                worker_class = getattr(module, "WORKER_CLASS")
                if worker_class:
                    if issubclass(worker_class, VCDWorkerInterface):
                        log.debug("[%s] Loaded class %s",
                                  file_basename, worker_class)
                    else:
                        log.debug("[%s] ``WORKER_CLASS`` value provided did "
                                  "not descend from VCDWorkerInterface. Not "
                                  "collecting.",
                                  file_basename)
                        worker_class = None
                else:
                    log.debug("[%s] ``WORKER_CLASS`` provided false-evaluating "
                              "value. Skipping module.", file_basename)
            else:
                log.debug("[%s] WORKER_CLASS not defined. Looking for class "
                          "matching module name...", file_basename)
                if (hasattr(module, file_basename)
                        and issubclass(getattr(module, file_basename),
                                       VCDWorkerInterface)):
                    log.debug("[%s] Worker class found via module name",
                              file_basename)
                    worker_class = getattr(module, file_basename)
                else:
                    log.debug("[%s] No class matching module name found. "
                              "Skipping module.", file_basename)

            if worker_class:
                # noinspection PyUnresolvedReferences
                # -> reason: must be a VCDWorkerInterface at this point
                module_map[worker_class.DESCRIPTOR_ID] = worker_class
                log.info('[%s] Loaded', file_basename)
            else:
                log.info('[%s] Not Loaded (see debug for why)', file_basename)

    return module_map
