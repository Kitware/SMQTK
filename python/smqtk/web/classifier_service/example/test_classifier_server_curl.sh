#!/bin/bash

run_echo() {
    (set -x; "$@")
}

expect() {
    echo "====>    Expecting" "$@"
}

# Basic read-only operations
run_echo curl -s -X GET localhost:5000/is_ready
expect "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo curl -s -X POST localhost:5000/classify \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Add a classifier with the same name as one configured by server
expect failure
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=dummy \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
expect "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Remove an immutable classifier
expect failure
run_echo curl -s -X DELETE localhost:5000/classifier \
    -d label=dummy
expect "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Add a new classifier under a new label
expect success
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
expect "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Remove a mutable classifier
expect success
run_echo curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
expect "dummy"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Add a new immutable classifier under a new label
expect success
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true  \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
expect "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Repeat the last operation (label now exists)
expect failure
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
expect "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Remove an immutable classifier
expect failure
run_echo curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
expect "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Retrieve a classifier under some label and store it
expect success
run_echo curl -s -X GET localhost:5000/classifier \
    -d label=foo > foo_classifier.pkl.b64
expect "dummy", "foo"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Add the classifier just retrieved under a new label
expect success
run_echo curl -s -X POST localhost:5000/classifier \
    -d label=bar \
    --data-urlencode bytes_b64@foo_classifier.pkl.b64
expect "dummy", "foo", "bar"
run_echo curl -s -X GET localhost:5000/classifier_labels
run_echo rm foo_classifier.pkl.b64
echo

# Retrieve a classifier under some nonexistent label
expect failure
run_echo curl -s -X GET localhost:5000/classifier \
    -d label=baz
expect "dummy", "foo", "bar"
run_echo curl -s -X GET localhost:5000/classifier_labels
echo

# Try to classify with a label we don't have
expect failure
run_echo curl -s -X POST localhost:5000/classify \
    -d label=baz \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with a label we have
expect success
run_echo curl -s -X POST localhost:5000/classify \
    -d label=bar \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with all of the labels we have, explicitly
expect success
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo","bar"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with some of the labels we have
expect success
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with none of the labels we have
expect success, but also empty result
run_echo curl -s -X POST localhost:5000/classify \
    -d label='[]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with some of the labels we have
expect success
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo

# Try to classify with some of the labels we have and some of the labels we
# don't
expect failure
run_echo curl -s -X POST localhost:5000/classify \
    -d label='["dummy","foo","baz"]' \
    -d "content_type=image/jpeg" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
echo
