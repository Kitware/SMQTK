#!/usr/bin/env python
# coding=utf-8
"""
This script is only called "europe" because of the specific bounding box below.
This bbox could be changed in a different region is desired.
"""

import logging
import pymongo

from masir.utils.webprivacy_gather import \
    SimpleBoundingBox,\
    wp_gather_image_and_info


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main():
    # x -> lon
    # y -> lat
    bbox = SimpleBoundingBox((22.1357201, 44.38643),
                             (40.2271720, 52.379475))
    db_coll = pymongo.MongoClient('mongo')['webprivacy']['tweet.ukraine']
    output_dir = "/home/purg/data/masir/webprivacy-gather-results/" \
                 "tweet.ukraine.ukraine_region"

    wp_gather_image_and_info(output_dir, db_coll, bbox)


if __name__ == "__main__":
    main()
