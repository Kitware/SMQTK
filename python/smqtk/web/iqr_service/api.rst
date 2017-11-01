Definitions
-----------

    session_id -> user identification string

    element_id -> SHA1 of image


.. _standard-message-base:

Standard Base Message Format
----------------------------

{
    message: <str>,
    time: {
        unix: <float>,
        utc: <str>
    },
    <other names params>
}


API End points
--------------
Here we document the various REST API end point available for use. The return
JSON data format is documented with an ellipses (``...``) for where the standard
return message content would be (:ref:`see above <standard-message-base>`).
All endpoints return a 200 or 201 status code upon successful operation.


[GET] /is_ready
^^^^^^^^^^^^^^^
Endpoint that response with a simple JSON message and response code 200 when the
server is active.

No arguments.

Returns 200.


[GET] /session_ids
^^^^^^^^^^^^^^^^^^
Get the list of current, active session IDs.

No arguments.

Returns: {
    ...
    session_uuids=<list[str]>
}


[GET] /session
^^^^^^^^^^^^^^
Get a JSON return with session state information.

This information includes the size of the session's working index
(``wi_count``) and the UUIDs, plus raw descriptor vectors, of positive and
negative examples. User provided pos/neg examples are separated out
(``uuids_pos_ext``, ``uuids_neg_ext``) from descriptors that are expected to
be a part of the service's configured backing descriptor set.

Form args:
    sid
        String session ID to get the information of.

Possible error code returns:
    400
        No session ID was provided.
    404
        The given session ID does not match a previously created session.

Returns 200: {
    ...
    sid=<session_id>,
    uuids_pos=<dict[str, list[float]]>
    uuids_neg=<dict[str, list[float]]>
    uuids_pos_ext=<dict[str, list[float]]>
    uuids_neg_ext=<dict[str, list[float]]>
    wi_count=<int>
}


[POST] /session
^^^^^^^^^^^^^^^
Creates a new session and returns the given or new SID.

Form Args:
    sid [optional]
        Explicit UUID to use for a new session. If not given we will generate a
        new UUID string and return it.

Possible error code returns:
    409
        Session ID provided already exists.

Returns 201: {
    ...
    sid=<session_id>
}


[PUT] /session
^^^^^^^^^^^^^^
Resets an existing session.

This does not remove the session from the controller, but just resets the
session's state. This means that adjudications, results and any classifiers
build for that session are cleared.

Form Args:
    sid
        Session ID (string) for the session.

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Returns 200: {
    ...
    sid=<session_id>
}


[DELETE] /session
^^^^^^^^^^^^^^^^^
Clear the resources associated with the given session id. The given session
id will not be usable until initialized again.

Form args:
    sid
        Session ID (string) for the session.

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Returns 200: {
    ...
    sid=<session_id>
}


[POST] /add_external_pos
^^^^^^^^^^^^^^^^^^^^^^^^
Describe the given data and consider the description as a positive exemplar from
external data for the given session, returning the UUID of the descriptor
generated.

Form args:
    sid
        The id of the session to add the generated descriptor to.
    base64
        The url-safe base64 byes of the data. This should use the same
        URL-safe alphabet as the python ``base64.urlsafe_b64decode``
        module function would expect.
    content_type
        The mimetype of data provided.

Possible error code returns:
    400
        No session ID provided. No or empty base64 data provided. No content
        mimetype provided.
    404
        No session for the given ID.

Returns 201: {
    ...
    descr_uuid=<str>
}


[POST] /add_external_neg
^^^^^^^^^^^^^^^^^^^^^^^^
Describe the given data and consider the description as a negative exemplar from
external data for the given session, returning the UUID of the descriptor
generated.

Form args:
    sid
        The id of the session to add the generated descriptor to.
    base64
        The url-safe base64 byes of the data. This should use the same
        URL-safe alphabet as the python ``base64.urlsafe_b64decode``
        module function would expect.
    content_type
        The mimetype of data provided.

Possible error code returns:
    400
        No session ID provided. No or empty base64 data provided. No content
        mimetype provided.
    404
        No session for the given ID.

Returns 201: {
    ...
    descr_uuid=<str>
}


[GET] /adjudicate
^^^^^^^^^^^^^^^^^
Get the adjudication state of a descriptor given its UID.

Arguments:
    sid
        Session ID.
    uid
        Descriptor UID to query for adjudication state.

Possible error code returns:
    400
        No session ID or descriptor UID provided.
    404
        No session for the given ID.
    500
        Descriptor labeled as both positive and negative somehow (indicates bug
        in server, should not be allowed possible).

Returns 200: {
    ...
    is_pos = <bool>
    is_neg = <bool>
}


[POST] /adjudicate
^^^^^^^^^^^^^^^^^^
Update the internal adjudication state given lists of new positive/negative
descriptor UIDs and now-neutral descriptor UIDs. All are optionally specified.
If nothing is provided in the parameters this functions logically does nothing.

If the same UID is present in both the positive and negative lists, they cancel
each other out and are considered neutral.

Changes to adjudications mark any current session classifier as dirty, requiring
a rebuilding of the session's classifier upon the next classification request.

Form Args:
    sid
        Session ID.
    pos
        List of descriptor UIDs that should be considered positive examples.
    neg
        List of descriptor UIDs that should be considered negative examples.
    neutral
        List of descriptor UIDs that should be considered neutral examples.

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Returns 200: {
    ...
    sid = <session_id>
}


[POST] /initialize
^^^^^^^^^^^^^^^^^^
Update the working index based on the current positive internal and external
descriptors.

This only updates the given session for positive descriptors that have not been
queried for before. Thus, if this endpoint is called twice in a row, the second
call should do nothing.

Form Args:
    sid
        Session ID

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Returns 200: {
    ...
    sid = <session_id>,
    success = <bool>
}


[POST] /refine
^^^^^^^^^^^^^^
Rank a session's working index based on the current positive and negative
adjudication state.

This sets or updated the results list for the given session.

Form Args:
    sid
        Session ID.

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Returns 201: {
    ...
    sid=<session_id>
}


[GET] /num_results
^^^^^^^^^^^^^^^^^^
Get number of results in the refined ranking list.

This is only non-zero after a refine operation has been performed on an
initialized working index. For example, this is 0 just after session
initialization or resetting.

Form Args:
    sid=<session_id>

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Returns 200: {
    ...
    sid=<session_id>,
    num_results=<int>
}


[GET] /get_results
^^^^^^^^^^^^^^^^^^
Get the ordered results between the optionally specified indices offset and
limit indices.

If ``i`` (offset, inclusive) is omitted, we assume a starting index of 0. If
``j`` (limit, exclusive) is omitted, we assume the ending index is the same as
the number of results available.

Return probability should be in the [0,1] range, where a value of 1.0 indicates
maximum relevance and 0.0 indicates the least relevance.

Form Args:
    sid
        Session ID.
    i [optional]
        Inclusive starting index in ordered results list (offset).
    j [optional]
        Exclusive end index in ordered results list (limit).

Returns 200: {
    ...
    sid=<session_id>,
    total_results=<int>,
    i=<int>,
    j=<int>,
    results=[(str:element_id, float:probability), ...]
}


[GET] /classify
^^^^^^^^^^^^^^^
Classify a number of descriptors based on the given list of descriptor UUIDs
based on the adjudication state of the current session.

A new classifier instance is built if there is no classifier already built for
the given session, or if the adjudication state has changed since the last time
the classifier was used.

This returns parallel ordered lists of the UUIDs of the given descriptors and
their positive classification probabilities. "Positive" in this classifier
is aligned with the positively adjudicated examples in the session.

Form Args:
    sid
        Session ID.
    uuids
        List of descriptor UUIDs to classify. These UUIDs must associate to
        descriptors in the configured descriptor index.

Possible error code returns:
    400
        - No session ID provided.
        - Failed to decode descriptor UUIDs list json provided.
        - No positive or negative adjudications for the given session (cannot
          build supervised classifier.
    404
        - No session for the given ID.
        - Could not find descriptors for at least one UUID provided.

Returns 200: {
    ...
    sid=<session_id>,
    uuids=[<element_id>, ...],
    proba=[<float>, ...],
}


[GET] /state
^^^^^^^^^^^^
Create and return a binary package representing this IQR session's state.

An IQR state is composed of the descriptor vectors, and their UUIDs, that were
added from external sources, or were adjudicated, positive and negative.

This endpoint directly returns the bytes of the created binary package in such a
form that it can be streamed to disk as a valid file (NOT in base64 and mostly
likely not URL-safe).

Arguments:
    sid
        Session ID to get the state of.

Possible error code returns:
    400
        No session ID provided.
    404
        No session for the given ID.

Success returns 200: {
    message = "Success"
    ...
    sid = <str>
    state_b64 = <str>
}


[PUT] /state
^^^^^^^^^^^^
Set the IQR session state for a given session ID.

We expect the input bytes to have been generated by the matching get-state
endpoint (see above).

Form Args:
    sid
        Session ID to set the input state to.
    state_base64
        Base64 of the state to set the session to.  This should be retrieved
        from the [GET] /state endpoint.

Possible error code returns:
    400
        - No session ID provided.
        - No base64 bytes provided.
    404
        No session for the given ID.

Success returns 200: {
    message = "Success"
    ...
    sid = <str>
}
