Definitions
-----------

    session_id -> user identification string

    element_id -> SHA1 of image


API End points
--------------

Standard return message: {
    message: <str>,
    time: {
        unix: <float>,
        utc: <str>
    },
    <other names params>
}

[POST] /session
    Creates a new session and returns given or new SID.

    Form Args:
        sid=<session_id>
            - Optional. If not given we will generate a new UUID string and
              return it.

    Returns: {
        ...
        sid=<session_id>
    }

[PUT] /session
    Resets an existing session. This basically means the results call will not
    return anything until a refine occurs again. Session ID is still usable
    after this call (resources not cleaned).

    Form Args:
        sid=<session_id>

    Returns {
        ...
        sid=<session_id>
    }

[DELETE] /session
    Clear the resources associated with the given session id. The given session
    id will not be usable until initialized again.

    Form args:
        sid=str

    Returns: {
        ...
        sid=<session_id>
    }

[PUT] /adjudicate
    Update the internal adjudication state given lists of new positive/negative
    UUIDs and now-neutral UUIDs. All are optionally specified.

    Form Args:
        sid=<session_id>
        pos=[element_id, ...]
        neg=[element_id, ...]
        neutral=[element_id, ...]

    Returns {
        ...
        sid=<session_id>
    }

[PUT] /refine
    Update working index and create/refine result ranking based on given
    positive and negative adjudication UUIDs. Adjudication given to this
    function is absolute (does not stack with previous refines/adjudications,
    but overwrites).

    (Flask SSE for progress status in the future?)

    Form Args:
        sid=<session_id>
        pos_uuids=[element_id, ...]
        neg_uuids=[element_id, ...]

    Returns: {
        ...
        sid=<session_id>
    }

[GET] /num_results
    Get total number in the refined ranking list. For example, this is 0 just
    after session initialization or resetting. In other terms, this is the size
    of the working index.

    Form Args:
        sid=<session_id>

    Returns: {
        ...
        sid=<session_id>,
        num_results=<int>
    }

[GET] /get_results
    Get ordered results between the optionally specified indices. If ``i`` is
    omitted, we assume a starting index of 0. If ``j`` is omitted, we assume the
    ending index is the size of the working index.

    Form Args:
        sid=<session_id>
        i=<int> [optional]
        j=<int> [optional]

    Return: {
        ...
        sid=<session_id>,
        total_results=<int>,
        i=<int>,
        j=<int>,
        results=[(element_id, float), ...]
    }

[GET] /classify
    Classify a given set of descriptors based on the adjudication state of the
    current session.

    Form Args:
        sid=<session_id>
        uuids=[<element_id>, ...]

    Return: {
        ...
        sid=<session_id>,
        uuids=[<element_id>, ...],
        proba=[<float>, ...],
    }
