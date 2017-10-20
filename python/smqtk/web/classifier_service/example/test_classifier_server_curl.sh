#!/bin/bash
set -x

curl -s -X GET localhost:5000/is_ready
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X POST localhost:5000/classifier \
    -d label=dummy \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X DELETE localhost:5000/classifier \
    -d label=dummy
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true  \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X POST localhost:5000/classifier \
    -d label=foo \
    -d lock_label=true \
    --data-urlencode bytes_b64@dummy_classifier.pkl.b64
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
curl -s -X DELETE localhost:5000/classifier \
    -d label=foo
curl -s -X GET localhost:5000/classifier_labels
curl -s -X POST localhost:5000/classify \
    -d "content_type=text/plain" \
    --data-urlencode bytes_b64@fish-bike.jpg.b64
