__author__ = "paul.tunison@kitware.com"


def merge_configs(a, b):
    """
    Merge configuration b into configuration a. Congruent keys are overwritten,
    while new keys are added.

    This is different than normal dictionary update in that we don't bash
    nested dictionaries, instead recursively updating them.

    """
    for k in b:
        if k in a and isinstance(a[k], dict) and isinstance(b[k], dict):
            merge_configs(a[k], b[k])
        else:
            a[k] = b[k]
