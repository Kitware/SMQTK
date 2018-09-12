import itertools


def check_empty_iterable(iterable, callback, exception_inst):
    """
    Check that the given iterable is not empty, then call the given callback
    function with the reconstructed iterable when it is not empty.

    :param iterable: Iterable to check.
    :type iterable: collections.Iterable

    :param callback: Function to call with the reconstructed, not-empty
        iterable.
    :type callback: (collections.Iterable) -> None

    :param exception_inst: The exception to throw if the iterable is empty
    :type exception_inst: Exception

    """
    i = iter(iterable)
    try:
        first = next(i)
    except StopIteration:
        raise exception_inst
    callback(itertools.chain([first], i))
