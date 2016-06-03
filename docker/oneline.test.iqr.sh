#!/usr/bin/env bash

# Use curl to test the IQR service.
# Assumes the use of the LEEDS Butterfly data set.
# Output should be visually verified

curl -X POST localhost:12346/session -d sid=0

# Mark positive some class 001 images, and one example of other classes as neg
curl -X PUT localhost:12346/refine \
    -d sid=0 \
    -d pos_uuids='["84f62ef716fb73586231016ec64cfeed82305bba", "a99fce7aa917c5756045e8c13c7815489fa10c25", "1f159256b37a4bc86ae79c43523f5d8c5e77fdef", "491d523ed4f159c96af6a3a03754a0919d67d16a", "41759b6ef37967227077471d1d12d7b9994f9e32"]' \
    -d neg_uuids='["2e56b2dda41c2bd734b789d8b4c1a4b253cdb414", "17bd8bbee8addc5f7f4b00b222824de240cea5eb", "a35586f14c2cb3571f8cd836ed394febc9f28a67", "8c5f0de4416ceefbc2281c901218c134868502dd", "f5fff08ba1a9340905434011688966e721f2463b", "b2d0b1d9c72ff32304bde06daac7f6c69fcd26c4", "02e057a67b5fe8705c2d95309333ca9bbea6d948", "01ad8b533b564c75e12180f88f9138e97e24ed7e"]'

curl -X GET localhost:12346/num_results?sid=0

curl -X GET localhost:12346/get_results?sid=0\&j=20
