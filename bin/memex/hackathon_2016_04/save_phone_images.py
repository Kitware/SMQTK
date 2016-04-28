#!/usr/bin/env python
"""
Intended to download annotated ad images.
Images stored under single directory, pathed similar to that as the URL it came from.
CDR ID relationships to images and truth associations maintained via CSV files.

This was used to unpack training data from the supplied CSV files.
"""
import collections
import itertools
import hashlib
import json
import logging
import mimetypes
import os
import requests
import StringIO
import uuid

from tika import detector as tika_detector

from smqtk.utils.bin_utils import initialize_logging
from smqtk.utils.file_utils import safe_create_dir
from smqtk.utils.parallel import parallel_map


initialize_logging(logging.getLogger('__main__'), logging.INFO)
initialize_logging(logging.getLogger('smqtk'), logging.INFO)
log = logging.getLogger(__name__)


if '.jfif' in mimetypes.types_map:
    del mimetypes.types_map['.jfif']
if '.jpe' in mimetypes.types_map:
    del mimetypes.types_map['.jpe']


def dl_ad_image(url, output_dir):
    """
    Returns (None, None, None) if failed, otherwise (url, filepath, sha1)
    """
    log = logging.getLogger(__name__)

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
    except requests.ConnectionError, ex:
        log.warn("Skipping '%s': %s: %s", url, str(type(ex)), str(ex))
        return None, None, None
    except requests.HTTPError, ex:
        log.warn("Skipping '%s': %s (code=%s)", url, r.reason, r.status_code)
        return None, None, None

    content = StringIO.StringIO()
    for c in r.iter_content(1024):
        content.write(c)
    cont_type = tika_detector.from_buffer(content.getvalue())
    ext = mimetypes.guess_extension(cont_type)
    if not ext:
        log.warn("Skipping '%s': Bad content type '%s'", url, cont_type)
        return None, None, None

    segs = url.split('/')
    dirpath = os.path.join(output_dir, *segs[2:-1])
    safe_create_dir(dirpath)

    basename = os.path.splitext(segs[-1])[0]
    save_pth = os.path.join(dirpath, basename + ext)

    if not os.path.isfile(save_pth):
        sha1_checksum = hashlib.sha1(content.getvalue()).hexdigest()
        tmp_pth = '.'.join([save_pth, uuid.uuid4().hex])
        with open(tmp_pth, 'wb') as f:
            f.write(content.getvalue())
        os.rename(tmp_pth, save_pth)
        log.info("Downloaded '%s' -> '%s'", url, save_pth)
    else:
        log.info("Already downloaded: '%s' -> '%s'", url, save_pth)
        with open(save_pth) as f:
            sha1_checksum = hashlib.sha1(f.read()).hexdigest()

    return url, save_pth, sha1_checksum


# Confirmed there are no conflicting truth labels on a CDR and URL basis
phone2urls_json = "/home/purg/dev/memex/hackathons/2016_04/cp1a/eval.source.phone2urls.json"
image_output_dir = "/home/purg/dev/memex/hackathons/2016_04/cp1a/eval.images"
# with open(ad_image_csv) as f:
#     url_ad_label_rows = list(csv.reader(f))
# with open(ad_phone_csv) as f:
#     ad2phone = dict(csv.reader(f))
with open(phone2urls_json) as f:
    phone2urls = json.load(f)

# Download unique img_urls, get filepaths + SHA1 checksum
url_set = set(url for p in phone2urls for url in phone2urls[p])
print "%d unique URLs" % len(url_set)
# URL to (filepath, sha1sum) tuple
#: :type: dict[str, (str, str)]
url2fs = {}
for url, save_pth, sha1 in parallel_map(dl_ad_image, url_set,
                                        itertools.repeat(image_output_dir),
                                        name='image_downloader',
                                        use_multiprocessing=True,
                                        # cores=32,
                                        cores=128,
                                        ):
    if url:
        url2fs[url] = (save_pth, sha1)
log.info("Downloaded %d images", len(url2fs))

log.info("Forming relational mappings")
#: :type: dict[str, str]
sha2path = {}
#: :type: dict[str, set[str]]
phone2shas = collections.defaultdict(set)

skipped = set()
included = set()
for phone, urls in phone2urls.iteritems():
    for url in urls:
        if url in url2fs:
            filepath, sha1 = url2fs[url]
            sha2path[sha1] = filepath
            phone2shas[phone].add(sha1)
            included.add(url)
        else:
            skipped.add(url)

log.info("Total files skipped: %d", len(skipped))
log.info("Total files included: %s", len(included))

log.info("Saving relational mappings as json")
json_opts = {
    'indent': 2,
    'separators': (',', ': ')
}

with open('eval.map.sha2path.json', 'w') as f:
    json.dump(sha2path, f, **json_opts)

with open('eval.map.phone2shas.json', 'w') as f:
    # convert set to list
    phone2shas = dict((p, list(shas)) for p, shas in phone2shas.iteritems())
    json.dump(phone2shas, f, **json_opts)
