#!/bin/bash
set -x

# Basic read-only operations
curl -s -X GET localhost:5000/is_ready
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Add a classifier with the same name as one configured by server
# Should fail
curl -s -X POST localhost:5000/classifier \
    -d label=dummy \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Remove an immutable classifier
# Should fail
curl -s -X DELETE localhost:5000/classifier \
    -d label=dummy
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Add a new classifier under a new label
# Should succeed
curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Remove a mutable classifier
# Should succeed
curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Add a new immutable classifier under a new label
# Should succeed
curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true  \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Repeat the last operation (label now exists)
# Should fail
curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Remove an immutable classifier
# Should fail
curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Retrieve a classifier under some label and store it
# Should succeed
curl -s -X GET localhost:5000/classifier \
    -d label=foo > foo_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64

# Add the classifier just retrieved under a new label
# Should succeed
curl -s -X POST localhost:5000/classifier \
    -d label=bar \
    --data-urlencode bytes_b64@foo_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
rm foo_classifier.pkl.b64

# Retrieve a classifier under some nonexistent label
# Should fail
curl -s -X GET localhost:5000/classifier \
    -d label=baz
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
