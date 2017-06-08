.. _standard-message-base:

Standard Base Message Format
----------------------------
All end points return a JSON response message of some kind. All returned JSON
shares the following base content. Specific endpoint may return additional
fields in place of the ellipses.

{
    message: <str>,
    time: {
        unix: <float>,
        utc: <str>
    },
    ...
}


End-point API
-------------
Below we document the various REST API end-points available for use with this
server. The return JSON data format is documented with an ellipse for where the
standard return message content would be
(:ref:`see above <standard-message-base>`).


[GET] /is_ready
^^^^^^^^^^^^^^^
Simple endpoint whose successful return just means this server is up and
responding.

No arguments.

Returns code 200 on success and the standard message
(:ref:`see above <standard-message-base>`).


[GET] /classifier_labels
^^^^^^^^^^^^^^^^^^^^^^^^
Get the descriptive labels of all (static and IQR-based) classifiers currently
available to classify input data.

Later, we describe an end-point that allows the creation of new label-classifier
pairs given IQR state file bytes.

No arguments.

Returns code 200 on success and the message: {
    ...
    labels: <list[str]>
}


[POST] /classify
^^^^^^^^^^^^^^^^
Given a file's bytes (standard base64-format) and content mimetype, describe and
classify the content against all currently stored classifiers, returning a map
of classifier descriptive labels to their class-to-probability results.

We expect the data to be transmitted in the body of the request in
standard base64 encoding form ("bytes_b64" key). We look for the content
type either as URL parameter or within the body ("content_type" key).

Data/Form arguments:
    bytes_base64
        Bytes in the standard base64 encoding to be described and classified.
    content_type
        The mimetype of the sent data.

Possible error codes:
    400
        No bytes or label provided

Returns code 200 on success and the message: {
    ...
    result: <dict[str, dict[str, float]]>
}


[GET] /iqr_classifier
^^^^^^^^^^^^^^^^^^^^^
Get the labels of the classifiers specifically added via uploaded IQR session
states.

No arguments.

Returns code 200 on success and the message: {
    ...
    labels: <list[str]>
}



[POST] /iqr_classifier
^^^^^^^^^^^^^^^^^^^^^^
Train a classifier based on the user-provided IQR state file bytes in standard
base64 encoding, matched with a descriptive label of that classifier's
topic.

Since all IQR session classifiers end up only having two result classes
(positive and negative), the topic of the classifier is encoded in the
descriptive label the user applies to the classifier.

Data/Form arguments:
    bytes_b64
        Bytes, in the standard base64 encoding, of the IQR session state save
        file.
    label
        Descriptive label to apply to this classifier. This should not
        conflict with existing classifier labels.

Possible error codes:
    400
        May mean one of:
            - No IQR state base64 data or label provided.
            - Label provided is in conflict with an existing label in the static
              or IQR classifier collections.

Returns code 201 on success and the message: {
    label: <str>
}


[DELETE] /iqr_classifier [optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Remove an IQR state classifier by the given label.

This end-point may or may not be enabled via the server configuration.

Data/Form arguments:
    label
        Descriptive label of the IQR state classifier to remove.

Possible error codes:
    400
        No label provided.
    404
        No IQR-state-based classifier to be removed for the given label.

Returns 200 on success and the message: {
    ...
    removed_label: <str>
}
