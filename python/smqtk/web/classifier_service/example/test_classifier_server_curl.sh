#!/bin/bash

run_echo() {
    (set -x; "$@")
}

# Basic read-only operations
run_echo curl -s -X GET localhost:5000/is_ready
echo :: Should have "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Add a classifier with the same name as one configured by server
echo :: Should fail
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=dummy \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
echo  :: Should have "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Remove an immutable classifier
echo :: Should fail
run_echo curl -s -X DELETE localhost:5000/classifier \
    -d label=dummy
echo  :: Should have "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Add a new classifier under a new label
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
echo  :: Should have "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Remove a mutable classifier
echo  :: Should succeed
run_echo curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
echo  :: Should have "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Add a new immutable classifier under a new label
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true  \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
echo  :: Should have "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Repeat the last operation (label now exists)
echo :: Should fail
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
echo  :: Should have "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Remove an immutable classifier
echo :: Should fail
run_echo curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
echo  :: Should have "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Retrieve a classifier under some label and store it
echo  :: Should succeed
run_echo curl -s -X GET localhost:5000/classifier \
    -d label=foo > foo_classifier.pkl.b64
echo  :: Should have "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Add the classifier just retrieved under a new label
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=bar \
    --data-urlencode bytes_b64@foo_classifier.pkl.b64
echo  :: Should have "dummy", "foo", "bar"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
run_echo rm foo_classifier.pkl.b64
echo

# Retrieve a classifier under some nonexistent label
echo :: Should fail
run_echo curl -s -X GET localhost:5000/classifier \
    -d label=baz
echo  :: Should have "dummy", "foo", "bar"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with a label we don't have
echo :: Should fail
run_echo curl -s -X POST localhost:5000/classify \
    -d label=baz \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with a label we have
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classify \
    -d label=bar \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with all of the labels we have, explicitly
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo","bar"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with some of the labels we have
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with none of the labels we have
echo  :: Should succeed, but have empty result
run_echo curl -s -X POST localhost:5000/classify \
    -d label='[]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with some of the labels we have
echo  :: Should succeed
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with some of the labels we have and some of the labels we
# don't
echo :: Should fail
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo","baz"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo
