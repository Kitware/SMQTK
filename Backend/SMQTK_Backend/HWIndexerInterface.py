"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

__hw_clip_ids = {}


def HWGlobalInitialize():
    return


def HWSessionInitialize(session_id, distance_kernel):
    """
    :param session_id:
    :type session_id: uuid.UUID
    :param distance_kernel:
    :type distance_kernel: DistanceKernel
    """
    global __hw_clip_ids
    distance_kernel.get_lock().acquireRead()
    __hw_clip_ids[session_id] = frozenset(distance_kernel.col_id_map())
    distance_kernel.get_lock().releaseRead()


def HWIndex(session_id, top_N_size, refined_positive, refined_negative):
    """
    :param session_id: The UUID of an initialized session.
    :type session_id: uuid.UUID
    :param top_N_size: The number of clips that are to have a high probability
        of being in the list of clip IDs that this function returns.
    :type top_N_size: int
    :param refined_negative: Iterable of clip IDs that are known to be positive.
    :type refined_negative: Iterable of int
    :param refined_positive: Iterable of clip IDs that are known to be negative.
    :type refined_positive: Iterable of int

    :raises KeyError: If session id given that wasn't initialized.

    :return: Pool of clip IDs that has a high probability of containing the top
        ``top_N_size`` videos. This dummy impl returns a number of IDs equal to
        5 times the given N size.
    :rtype: tuple of int

    """
    ret_pool_size = top_N_size * 5
    ordered_id_list = \
        __hw_clip_ids[session_id].intersection(refined_positive).union(
            __hw_clip_ids[session_id].difference(refined_positive, refined_negative).union(
                __hw_clip_ids[session_id].intersection(refined_negative)))
    ordered_id_list = sorted(tuple(ordered_id_list))  # pesky sets...
    assert len(ordered_id_list) == len(__hw_clip_ids[session_id])
    return ordered_id_list[:ret_pool_size]
