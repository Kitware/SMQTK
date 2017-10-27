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
Given a file's bytes (standard base64-format) and content mimetype, describe
and classify the content against all currently stored classifiers (optionally
a list of requested classifiers), returning a map of classifier descriptive
labels to their class-to-probability results.

We expect the data to be transmitted in the body of the request in
standard base64 encoding form ("bytes_b64" key). We look for the content
type either as URL parameter or within the body ("content_type" key).

Below is an example call to this endpoint via the ``requests`` python
module, showing how base64 data is sent::

    import base64
    import requests
    data_bytes = "Load some content bytes here."
    requests.post('http://localhost:5000/classify',
                  data={'bytes_b64': base64.b64encode(data_bytes),
                        'content_type': 'text/plain'})

With curl on the command line::

    $ curl -X POST localhost:5000/classify -d "content_type=text/plain" \
        --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

    # If this fails, you may wish to encode the file separately and use the
    # file reference syntax instead:

    $ base64 -w0 /path/to/file > /path/to/file.b64
    $ curl -X POST localhost:5000/classify -d "content_type=text/plain" \
        --data-urlencode bytes_64=@/path/to/file.b64

Optionally, the `label` parameter can be provided to limit the results of
classification to a set of classifiers::

    $ curl -X POST localhost:5000/classify -d "content_type=text/plain" \
        -d 'label=["some_label","other_label"]' \
        --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

    # If this fails, you may wish to encode the file separately and use the
    # file reference syntax instead:

    $ base64 -w0 /path/to/file > /path/to/file.b64
    $ curl -X POST localhost:5000/classify -d "content_type=text/plain" \
        -d 'label=["some_label","other_label"]' \
        --data-urlencode bytes_64=@/path/to/file.b64

Data/Form arguments:
    bytes_b64
        Bytes in the standard base64 encoding to be described and classified.
    content_type
        The mimetype of the sent data.
    label
        (Optional) Label of the requested classifier, or JSON list of
        requested classifiers

Possible error codes:
    400
        No bytes provided
    404
        Label or labels provided do not match any registered classifier

Returns code 200 on success and the message: {
    ...
    result: <dict[str, dict[str, float]]>
}


[GET] /iqr_classifier [defunct]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DEFUNCT: use [GET] /classifier_labels instead.

Get the descriptive labels of all (static and IQR-based) classifiers currently
available to classify input data. (Previously returned only IQR-based
classifiers. This service no longer distinguishes between IQR and static
classifiers.)

No arguments.

Returns code 200 on success and the message: {
    ...
    labels: <list[str]>
}


[GET] /classifier
^^^^^^^^^^^^^^^^^
Download the classifier corresponding to the provided label, pickled and
encoded in standard base64 encoding.

Below is an example call to this endpoint via the ``requests`` python module::

    import base64
    import requests
    from six.moves import cPickle as pickle

    r = requests.get('http://localhost:5000/classifier',
                     data={'label': 'some_label'})
    data_bytes = base64.b64decode(r.content)
    classifier = pickle.loads(data_bytes)

With curl on the command line::

    $ curl -X GET localhost:5000/classifier -d label=some_label | \
        base64 -d > /path/to/file.pkl

Data args:
    label
        Label of the requested classifier

Possible error codes:
    400
        No label provided
    404
        Label does not refer to a registered classifier

Returns 200 on success and the pickled and encoded classifier as the response
content


[POST] /classifier
^^^^^^^^^^^^^^^^^^
Upload a **trained** classifier pickled and encoded in standard base64
encoding, matched with a descriptive label of that classifier's topic.

The topic of the classifier is encoded in the descriptive label the user
applies to the classifier.

Below is an example call to this endpoint via the ``requests`` python
module, showing how base64 data is sent::

    import base64
    import requests
    from six.moves import cPickle as pickle

    classifier = None # Instantiate a classifier
    data_bytes = pickle.dumps(classifier)
    requests.post('http://localhost:5000/classifier',
                  data={'bytes_b64': base64.b64encode(data_bytes),
                        'label': 'some_label'})

With curl on the command line::

    $ curl -X POST localhost:5000/classifier -d label=some_label \
        --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file.pkl)"

    # If this fails, you may wish to encode the file separately and use the
    # file reference syntax instead:

    $ base64 -w0 /path/to/file.pkl > /path/to/file.pkl.b64
    $ curl -X POST localhost:5000/classifier -d label=some_label \
        --data-urlencode bytes_64=@/path/to/file.pkl.b64

To lock this classifier and guard it against deletion, add "lock_label=true"::

    $ curl -X POST localhost:5000/classifier \
        -d "label=some_label" \
        -d "lock_label=true" \
        --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file.pkl)"

Data/Form arguments:
    bytes_b64
        Bytes, in the standard base64 encoding, of the pickled classifier.
    label
        Descriptive label to apply to this classifier. This should not
        conflict with existing classifier labels.
    lock_label
        If 'true', disallow deletion of this label. If 'false', allow
        deletion of this label. Only has an effect if deletion is
        enabled for this service. (Default: 'false')

Possible error codes:
    400
        May mean one of:
            - No pickled classifier base64 data or label provided.
            - Label provided is in conflict with an existing label in the
              classifier collection.

Returns code 201 on success and the message: {
    label: <str>
}


[POST] /iqr_classifier
^^^^^^^^^^^^^^^^^^^^^^
**Train** a classifier based on the user-provided IQR state file bytes in
standard base64 encoding, matched with a descriptive label of that
classifier's topic.

Since all classifiers have only two result classes (positive and negative),
the topic of the classifier is encoded in the descriptive label the user
applies to the classifier.

Below is an example call to this endpoint via the ``requests`` python
module, showing how base64 data is sent::

    import base64
    import requests
    data_bytes = "Load some content bytes here."
    requests.post('http://localhost:5000/iqr_classifier',
                  data={'bytes_b64': base64.b64encode(data_bytes),
                        'label': 'some_label'})

With curl on the command line::

    $ curl -X POST localhost:5000/iqr_classifier -d label=some_label \
        --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

    # If this fails, you may wish to encode the file separately and use the
    # file reference syntax instead:

    $ base64 -w0 /path/to/file > /path/to/file.b64
    $ curl -X POST localhost:5000/iqr_classifier -d label=some_label \
        --data-urlencode bytes_64=@/path/to/file.b64

To lock this classifier and guard it against deletion, add "lock_label=true"::

    $ curl -X POST localhost:5000/iqr_classifier \
        -d "label=some_label" \
        -d "lock_label=true" \
        --data-urlencode "bytes_b64=$(base64 -w0 /path/to/file)"

Data/Form arguments:
    bytes_b64
        Bytes, in the standard base64 encoding, of the IQR session state save
        file.
    label
        Descriptive label to apply to this classifier. This should not
        conflict with existing classifier labels.
    lock_label
        If 'true', disallow deletion of this label. If 'false', allow
        deletion of this label. Only has an effect if deletion is
        enabled for this service. (Default: 'false')

Possible error codes:
    400
        May mean one of:
            - No IQR state base64 data or label provided.
            - Label provided is in conflict with an existing label in the
              classifier collection.

Returns code 201 on success and the message: {
    label: <str>
}


[DELETE] /classifier [optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Remove a classifier by the given label.

This end-point may or may not be enabled via the server configuration.

Data/Form arguments:
    label
        Descriptive label of the classifier to remove.

Possible error codes:
    400
        No label provided.
    404
        No classifier to be removed for the given label.

Returns 200 on success and the message: {
    ...
    removed_label: <str>
}


[DELETE] /iqr_classifier [defunct]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DEFUNCT: Use [DELETE] /classifier instead.

Remove a classifier by the given label. (Previously deleted only an IQR-based
classifier. This service no longer distinguishes between IQR and static
classifiers.)

This end-point may or may not be enabled via the server configuration.

Data/Form arguments:
    label
        Descriptive label of the classifier to remove.

Possible error codes:
    400
        No label provided.
    404
        No classifier to be removed for the given label.

Returns 200 on success and the message: {
    ...
    removed_label: <str>
}

