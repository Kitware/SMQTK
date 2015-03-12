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
    bbox = SimpleBoundingBox((-12.061411, 35.897053),
                             (41.288200, 71.388295))
    db_coll = pymongo.MongoClient('mongo')['webprivacy']['tweet']
    output_dir = "/home/purg/data/masir/webprivacy-gather-results/" \
                 "full_set.europe"

    wp_gather_image_and_info(output_dir, db_coll, bbox)


if __name__ == "__main__":
    main()
