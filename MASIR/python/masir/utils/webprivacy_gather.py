# coding=utf-8

import bson
import datetime
import logging
import os
import urllib2

from masir.utils.SimpleBoundingBox import SimpleBoundingBox


def wp_gather_image_and_info(output_dir, db_collection, region_bbox):
    """
    Gather the imagery and metadata (as BSON) from the webprivacy database to an
    output directory given date and region constraints.

    :param output_dir: Directory to write files
    :type output_dir: str
    :param db_collection: pymongo database collection object
    :type db_collection: pymongo.collection.Collection
    :param region_bbox: Geographic region constraint.
    :type region_bbox: SimpleBoundingBox

    """
    log = logging.getLogger('gather_tweet_image_and_info')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log.info("Performing Query")
    after = datetime.datetime(2014, 2, 1)
    q = db_collection.find({
        "$and": [
            {"media": {"$gt": {"$size": 0}}},
            {"media.0.type": "photo"},
            {"created": {"$gte": after}},
            {"geo.0": {"$gte": region_bbox.min_x, "$lte": region_bbox.max_x}},
            {"geo.1": {"$gte": region_bbox.min_y, "$lte": region_bbox.max_y}},
        ]
    }, timeout=False)

    log.info("Scanning results")
    count = 0
    try:
        for i, e in enumerate(q):
            log.info("[%9d] %s -> %s %s", i, e['_id'], e['geo'],
                     e['media'][0]['media_url'])
            media_path = e['media'][0]['media_url']
            media_filetype = os.path.splitext(media_path)[1]

            output_image = os.path.join(output_dir, str(e['_id']) + media_filetype)
            output_bson = os.path.join(output_dir, str(e['_id']) + ".bson")

            try:
                with open(output_image, 'wb') as output_image_file:
                    output_image_file.write(urllib2.urlopen(media_path).read())
                with open(output_bson, 'wb') as output_md_file:
                    output_md_file.write(bson.BSON.encode(e))
                count += 1
            except Exception, ex:
                # Skip all files that cause errors anywhere. Remove anything
                # written
                log.info("     - ERROR: %s", str(ex))
                log.info("     - Skipping")
                if os.path.exists(output_image):
                    os.remove(output_image)
                if os.path.exists(output_bson):
                    os.remove(output_bson)
    finally:
        # Since we checked out a cursor without a server-side timeout, make sure
        # that we catch whatever and close the cursor when we're done / if
        # things fail.
        q.close()

    log.info("Discovered %d matching entries with valid images.", count)