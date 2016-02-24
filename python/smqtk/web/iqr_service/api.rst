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

[PUT] /init_session
    Creates a new session and returns given or new SID.

    Form Args:
        session_id=<str>
            - Optional. If not given we will generate a new UUID string and
              return it.

    Returns: {
        ...
        sid=<session_id>
    }

[PUT] /reset_session
    Resets an existing session. This basically means the results call will not
    return anything until a refine occurs again. Session ID is still usable
    after this call (resources not cleaned).

    Form Args:
        session_id=<str>

    Returns {
        ...
        sid=<session_id>
    }

[PUT] /clean_session
    Clear the resources associated with the given session id. The given session
    id will not be usable until initialized again.

    Form args:
        session_id=str

    Returns: {
        ...
        sid=<session_id>
    }

[PUT] /refine
    Update working index and create/refine result ranking based on given
    positive and negative adjudication UUIDs. (Flask SSE for progress status in
    the future?).

    Form Args:
        session_id=<str>
        pos=[element_id, ...]
        neg=[element_id, ...]

    Returns: {
        ...
        sid=<session_id>
    }

[GET] /num_results
    Get total number in the refined ranking list. For example, this is 0 just
    after session initialization or resetting. In other terms, this is the size
    of the working index.

    Form Args:
        session_id=<str>

    Returns: {
        ...
        sid=<session_id>,
        num_results=<int>
    }

[GET] /get_results(session_id:str, i:int, j:int) -> list[element_id:str]
    Get ordered results between the optionally specified indices. If ``i`` is
    omitted, we assume a starting index of 0. If ``j`` is omitted, we assume the
    ending index is the size of the working index.

    Form Args:
        session_id=<str>
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
