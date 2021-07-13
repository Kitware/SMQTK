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


Transmitting Base64 data
^^^^^^^^^^^^^^^^^^^^^^^^
Various PUT and POST endpoints accept base64 encoded data.  The following
examples show how to provide base64 data to an endpoint via the python
``requests`` module as well as curl.

Python ``requests``::

    import base64
    import requests
    data_bytes = "Load some data bytes here. Make sure you know the content "
                 "type of the bytes. Lets pretend this is image bytes."
    data_content_type = "image/jpeg"
    requests.post('http://localhost:5000/some_post_endpoint',
                  data={'base64': base64.b64encode(data_bytes),
                        'content_type': data_content_type})

With curl on the command line, we have two options.  If the bytes of the data
being sent is not too long, we can encode the data in-line with the curl
command::

    $ curl -X POST localhost:5000/some_post_endpoint \
        -d content_type=image/jpeg \
        --data-urlencode "data_b64=$(base64 -w0 /path/to/file)"

If the data file is too long for command-line expansion (usually the case)::

    $ # Encode the target file separately.
    $ base64 -w0 /path/to/file >/path/to/file.b64
    $ # Tell curl to read from the file instead of expanding in-line.
    $ curl -X POST localhost:5000/some_post_endpoint \
        -d content_type=image/jpeg \
        --data-urlencode data_b64@/path/to/file.b64


[GET] /is_ready
^^^^^^^^^^^^^^^
Endpoint that response with a simple JSON message and response code 200 when the
server is active.

No arguments.

Returns 200.


[POST] /add_descriptor_from_data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Add the description of the given base64 data with content type to the
descriptor set.

Accept base64 data (with content type), describe it via the configured
descriptor generator and add the resulting descriptor element to the
configured descriptor index.

Form Arguments:
    data_b64
        Base64-encoded input binary data to describe via
        DescriptorGenerator.  This must be of a content type accepted by
        the configured DescriptorGenerator.
    content_type
        Input data content mimetype string.

Success returns 201: {
    ...,

    // UID of the descriptor element generated from input data
    // description.  This should be equivalent to the SHA1 checksum of
    // the input data.
    uid <str>

    // New size (integer) of the descriptor set that has been updated
    // (NOT the same as the nearest-neighbor index).
    size <int>
}


[GET] /nn_index
^^^^^^^^^^^^^^^
Get status/state information about the nearest-neighbor index.

Status code 200 on success, JSON return object: {
    ...,
    // Size of the nearest-neighbor index.
    index_size=<int>
}


[POST] /nn_index
^^^^^^^^^^^^^^^^
Tell the configured nearest-neighbor-index instance to update with the
descriptors associated with the provided list of UIDs.

This is a critical operation on the index so this method can only be
invoked once at a time (other concurrent will block until previous calls
have finished).

Form Arguments:
    descriptor_uids
        JSON list of UID strings.  If one or more UIDs do not match
        descriptors in our current descriptor-set we return an error
        message.

Success returns 200: {
    ...,

    // List of UIDs the neighbor index was updated with.  This should
    // be congruent with the list provided.
    descriptor_uids=<list[str]>,

    // New size of the nearest-neighbors index.
    index_size=<int>
}


[DELETE] /nn_index
^^^^^^^^^^^^^^^^^^
Remove descriptors from the nearest-neighbors index given their UIDs.

Receive one or more descriptor UIDs, that exist in the NN-index, that
are to be removed from the NN-index.  This DOES NOT remove elements
from the global descriptor set.

This is a critical operation on the index so this method can only be
invoked once at a time (other concurrent will block until previous
calls have finished).

Form Arguments:
    descriptor_uids
        JSON list of descriptor UIDs to remove from the nearest-
        neighbor index.  These UIDs must be present in the index,
        otherwise an 404 error is returned.

Status code 200 on success, JSON return object: {
    ...,

    // List of UID values removed from the index.
    descriptor_uids=<list[str]>,

    // New size of the nearest-neighbors index.
    index_size=<int>
}


[POST] /data_nearest_neighbors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Take in data in base64 encoding with a mimetype and find its 'k' nearest
neighbors according to the current index, including their distance
values (metric determined by nearest-neighbors-index algorithm
configuration).

This endpoint does not need a session ID due to the
nearest-neighbor-index being a shared resource across IQR sessions.

Form Arguments:
    data_b64
        Base64-encoded input binary data to describe via
        DescriptorGenerator.  This must be of a content type accepted by
        the configured DescriptorGenerator.
    content_type
        Input data content mimetype string.
    k
        Integer number of nearest neighbor descriptor UIDs to return
        along with their distances.

Success returns 200: {
    ...,

    // Ordered list of neighbor UID values. Index 0 represents the
    // closest neighbor while the last index represents the farthest
    // neighbor.  Parallel in relationship to `neighbor_dists`.
    neighbor_uids=<list[str]>,

    // Ordered list of neighbor distance values. Index 0 represents the
    // closest neighbor while the last index represents the farthest
    // neighbor.  Parallel in relationship to 'neighbor_uids`.
    neighbor_dists=<list[float]>
}


[GET] /uid_nearest_neighbors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Take in the UID that matches an ingested descriptor and find that
descriptor's 'k' nearest neighbors according to the current index,
including their distance values (metric determined by
nearest-neighbors-index algorithm configuration).

This endpoint does not need a session ID due to the
nearest-neighbor-index being a shared resource across IQR sessions.

This endpoint can be more advantageous compared the
`data_nearest_neighbors` endpoint if you know a descriptor has already
been ingested (via `add_descriptor_from_data` or otherwise) as a
potentially new descriptor does not have to be computed.

URL Arguments:
    uid
        UID of the descriptor to get the nearest neighbors for.  This
        should also match the SHA1 checksum of the data being described.
    k
        Integer number of nearest neighbor descriptor UIDs to return
        along with their distances.

Success returns 200: {
    ...,

    // Ordered list of neighbor UID values. Index 0 represents the
    // closest neighbor while the last index represents the farthest
    // neighbor.  Parallel in relationship to `neighbor_dists`.
    neighbor_uids=<list[str]>,

    // Ordered list of neighbor distance values. Index 0 represents the
    // closest neighbor while the last index represents the farthest
    // neighbor.  Parallel in relationship to 'neighbor_uids`.
    neighbor_dists=<list[float]>,
}


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
Get the relevancy score for working index descriptor elements between
the optionally specified offset and limit indices, ordered by
*descending* predicted relevancy values (in [0, 1] range).

If ``i`` (offset, inclusive) is omitted, we assume a starting index of
0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
is the same as the number of results available.

If the requested session has not been refined yet (no ranking), an
empty results list is returned.

URL Args:
    sid: str
        UUID of the session to use
    i: int
        Starting index (inclusive)
    j: int
        Ending index (exclusive)

Possible error code returns:
    400
        No session ID provided. Offset/limit index values were not
        valid integers.
    404
        No session for the given ID.

Success returns 200 and a JSON object that includes the keys:
    sid: str
        String IQR session ID accessed.
    i: int
        Index offset used.
    j: int
        Index limit used.
    total_results: int
        Total number of ranked results with predicted relevancy. This
        is not necessarily the number of results returned from the
        call due to the optional use of ``i``  and ``j``.
    results: list[(str, float)]
        A list of ``(element_id, probability)`` pairs. The
        ``element_id`` is the UUID of the data/descriptor the result
        relevancy probability score is associated do. The
        ``probability`` value is a float in the [0, 1] range.

[GET] /get_feedback
^^^^^^^^^^^^^^^^^^
Get the feedback results for working set descriptor elements that are
recommended for adjudication feedback. They are listed with the most
useful first.

If the requested session has not been refined yet (no ranking), an
empty results list is returned.

URL Args:
    sid: str
        UUID of the session to use
    i: int
        Starting index (inclusive)
    j: int
        Ending index (exclusive)

Possible error code returns:
    400
        No session ID provided. Offset/limit index values were not
        valid integers.
    404
        No session for the given ID.

Success returns 200 and a JSON object that includes the keys:
    sid: str
        String IQR session ID accessed.
    i: int
        Index offset used.
    j: int
        Index limit used.
    total_results: int
        Total number of feedback results. This is not necessarily the
        number of results returned from the call due to the optional
        use of ``i``  and ``j``.
    results: list[str]
        A list of ``element_ids``. The ``element_id`` is the UUID of
        the data/descriptor that is recommending for feedback.

[GET] /get_positive_adjudication_relevancy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Get the relevancy scores for positively adjudicated elements in the
working index between the optionally provided index offset and limit,
ordered by *descending* predicted relevancy values (in [0, 1] range).

If ``i`` (offset, inclusive) is omitted, we assume a starting index of
0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
is the same as the number of results available.

If the requested session has not been refined yet (no ranking), an
empty results list is returned.

URI Args:
    sid: str
        UUID of the IQR session to use.
    i: int
        Starting index (inclusive).
    j: int
        Ending index (exclusive).

Possible error code returns:
    400
        No session ID provided. Offset/limit index values were not
        valid integers.
    404
        No session for the given ID.

Returns 200 and a JSON object that includes the following:
    sid: str
        String IQR session ID accessed.
    i: str
        Index offset used.
    j: str
        Index limit used.
    total: int
        Total number of positive adjudications in the current IQR
        session.
    results: list[(str, float)]
        List of ``(uuid, score)`` tuples for positively adjudicated
        descriptors in the working index, ordered by descending score.


[GET] /get_negative_adjudication_relevancy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Get the relevancy scores for negatively adjudicated elements in the
working index between the optionally provided index offset and limit,
ordered by *descending* predicted relevancy values (in [0, 1] range).

If ``i`` (offset, inclusive) is omitted, we assume a starting index of
0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
is the same as the number of results available.

If the requested session has not been refined yet (no ranking), an
empty results list is returned.

URI Args:
    sid: str
        UUID of the IQR session to use.
    i: int
        Starting index (inclusive).
    j: int
        Ending index (exclusive).

Possible error code returns:
    400
        No session ID provided. Offset/limit index values were not
        valid integers.
    404
        No session for the given ID.

Returns 200 and a JSON object that includes the following:
    sid: str
        String IQR session ID accessed.
    i: int
        Index offset used.
    j: int
        Index limit used.
    total: int
        Total number of negative adjudications in the current IQR
        session.
    results: list[(str, float)]
        List of ``(uuid, score)`` tuples for negatively adjudicated
        descriptors in the working index, ordered by descending score.


[GET] /get_unadjudicated_relevancy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Get the relevancy scores for non-adjudicated elements in the working
index between the optionally provided index offset and limit, ordered
by descending predicted relevancy value ([0, 1] range).

If ``i`` (offset, inclusive) is omitted, we assume a starting index of
0. If ``j`` (limit, exclusive) is omitted, we assume the ending index
is the same as the number of results available.

If the requested session has not been refined yet (no ranking), an
empty results list is returned.

URI Args:
    sid: str
        UUID of the IQR session to use.
    i: int
        Starting index (inclusive).
    j: int
        Ending index (exclusive).

Possible error code returns:
    400
        No session ID provided. Offset/limit index values were not
        valid integers.
    404
        No session for the given ID.

Returns 200 and a JSON object that includes the following:
    sid: str
        String IQR session ID accessed.
    i: int
        Index offset used.
    j: int
        Index limit used.
    total: int
        Total number of negative adjudications in the current IQR
        session.
    results: list[(str, float)]
        List of ``(uuid, score)`` tuples for negatively adjudicated
        descriptors in the working index, ordered by descending score.


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
        - No UIDs provided.
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
