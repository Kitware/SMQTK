# coding=utf-8

import bson
import copy
import hashlib
import logging
import multiprocessing
import os
import os.path as osp
import PIL.Image
import re
import shutil


def touch(fname):
    with open(fname, 'a'):
        os.utime(fname, None)


class IngestItemInvalidMetadata (Exception):
    pass


class IngestItemMetadata (object):
    """
    Metadata paired with an ingest.

    Constructed with a dict object with minimal checks for what is contained
    within allowing for extra information to be contained that is not
    necessarily expected.

    """

    @classmethod
    def from_file(cls, filepath):
        """
        Construct an IngestItemMetadata object from a given filepath pointing to
        a BSON data file.

        :raises InvalidBSON: The given file is not a valid BSON data file
        :raises AssertionError: The metadata contained in the BSON data was
            invalid (did not contain required fields).

        :returns: Constructed IngestItemMetadata
        :rtype: IngestItemMetadata

        """
        try:
            with open(filepath) as infile:
                return cls(bson.BSON(infile.read()).decode())
        except AssertionError:
            raise IngestItemMetadata(
                "Given bson file does not container required metadata "
                "components: %s" % filepath
            )

    def __init__(self, d):
        """
        Initialize metadata object with a dictionary
        """
        assert (
            isinstance(d, dict) and d

            and 'media' in d
            and d['media']
            and 'media_url' in d['media'][0]

            and 'geo' in d
            and d['geo']

        ), "[IngestItemMetadata] Failed metadata content assertion"
        self._d = d

    def __getitem__(self, item):
        return self._d[item]

    def as_dict(self):
        """
        Return the metadata in the raw dictionary form.

        This returns a copy, so modifications of returned dictionary does not
        affect this object's stored contents.
        """
        return copy.deepcopy(self._d)

    def as_json_dict(self):
        """
        Return a JSON compliant dictionary of metadata elements

        NOTE: This is pretty manual according to the format at the moment due
              to needing to hand massage some data-types. Could make a tree
              parser where the parsing decision is based on an element's type,
              but that's a little more complicated that we need at the moment
              since there is a single assumed format.

        """
        j = self.as_dict()
        j['_id'] = str(j['_id'])
        j['created'] = str(['created'])
        return j

    def get_web_image_address(self):
        """
        Return internet URL of original image.
        """
        return self._d['media'][0]['media_url']

    def get_geo_loc(self):
        """
        Return geographic location in a (lon, lat) tuple.
        """
        return tuple(self._d['geo'])


class IngestManager (object):
    """
    Manages ingested files

    Ingested file names are of the format: <md5SumHex>.uid_<idnumber>.<ext>

    Supports an ingest size of 999,999,999 files (may be increased by modifying
    FILE_REGEX property before object construction).

    Supports an optional explicit list detailing which IDs are to be considered
    explicit images. We expect this file to be a line separated list of image
    IDs.

    """

    # Standard ingested file regular expression
    # Groups:
    #   1: md5 sum (hex)
    #   2: integer unique ID
    FILE_REGEX = re.compile("(\w+)\.uid_(\d{9})\..*")

    # string template for ingested files. Takes: (md5, id, ext)
    INGEST_FILE_TMPL = "%s.uid_%09d%s"

    INGEST_SUBDIR_IMAGERY = 'image_ingest'
    INGEST_SUBDIR_METADATA = "metadata_ingest"

    def __init__(self, base_data_dir, starting_id=0):
        base_data_dir = osp.abspath(osp.expanduser(base_data_dir))
        self._image_data_dir = osp.join(base_data_dir,
                                        self.INGEST_SUBDIR_IMAGERY)
        self._metadata_data_dir = osp.join(base_data_dir,
                                           self.INGEST_SUBDIR_METADATA)

        if not osp.isdir(self._image_data_dir):
            os.makedirs(self._image_data_dir)
        if not osp.isdir(self._metadata_data_dir):
            os.makedirs(self._metadata_data_dir)

        # file recording which IDs are considered explicit
        self._explicit_ids_list_file = osp.join(base_data_dir,
                                                'image_ingest_explicit_ids.txt')
        if not osp.isfile(self._explicit_ids_list_file):
            touch(self._explicit_ids_list_file)
        self._explicit_ids = set()
        self._eid_mod_lock = multiprocessing.RLock()

        self._starting_id = starting_id

        # Mapping of image ID to the image file's path (post-ingestion)
        #: :type: dict of (int, str)
        self._id_path_map = {}

        # Mapping of image ID to the metadata dictionary associated to that file
        #: :type: dict of (int, dict)
        self._id_metadata_map = {}

        # Mapping of an MD5 object to the ID (+ inverse). Mainly used for
        # testing duplicate ingest.
        #: :type: dict of (hashlib.md5, list of int)
        self._md5_id_map = {}

        # Check for existing ingested files, accumulating metadata
        self._check_ingested()

    def __len__(self):
        return len(self._id_path_map)

    @property
    def image_data_dir(self):
        """ Read-only data directory property """
        return self._image_data_dir

    @property
    def metadata_data_dir(self):
        return self._metadata_data_dir

    @property
    def _log(self):
        return logging.getLogger('.'.join((self.__module__,
                                           self.__class__.__name__)))

    @property
    def ingest_md5(self):
        """ MD5 sum of entire ingest

        Based on sorted file MD5 sums for deterministic results.

        :return: Hexadecimal MD5 sum of this ingest
        :rtype: str

        """
        return hashlib.md5(str(sorted(
            [(md5, len(id_list)) for md5, id_list in self._md5_id_map.items()]
        ))).hexdigest()

    def _check_ingested(self):
        """
        Check configured data directory for ingested files, updating the id map
        with files found. We assume that all files we find here are valid image
        files.
        """
        for img_file in os.listdir(self.image_data_dir):
            img_filepath = osp.join(self.image_data_dir, img_file)
            m = self.FILE_REGEX.match(img_file)
            if m:
                md5 = m.group(1)
                uid = int(m.group(2))

                self._id_path_map[uid] = img_filepath
                self._md5_id_map.setdefault(md5, []).append(uid)

                # load in paired metadata file if there is one
                md_filepath = \
                    osp.join(self._metadata_data_dir,
                             self.INGEST_FILE_TMPL % (md5, uid, ".bson"))
                if osp.exists(md_filepath):
                    self._id_metadata_map[uid] = \
                        IngestItemMetadata.from_file(md_filepath)
            else:
                raise ValueError("Encountered ingested image file with invalid "
                                 "filename "
                                 "(does not match expect regular "
                                 "expression: \"%s\") "
                                 "(file in question: \"%s\")"
                                 % (self.FILE_REGEX.pattern, img_file))

        if osp.isfile(self._explicit_ids_list_file):
            with self._eid_mod_lock:
                with open(self._explicit_ids_list_file, 'r') as elfile:
                    for line in elfile.readlines():
                        self._explicit_ids.add(int(line.strip()))

    def ingest_image(self, image_filepath, metadata_filepath=None,
                     is_explicit=False):
        """
        Ingest an image file.

        The given file is copied into our ingest so it is safe to delete the
        file at the given input path after this function returns.

        :raises ValueError: The given file at the provided path has already been
            ingested (based on MD5 checksum).
        :raises IOError: Not given a valid image file.
        :raises InvalidBSON: Not given a valid metadata file if one was given.

        :param image_filepath: Path to the image file to ingest
        :type image_filepath: str
        :param metadata_filepath: Optional paired metadata fail path associated
            with the given image file. This file must be BSON encoded.
        :type metadata_filepath: str or None
        :param is_explicit: Whether or not this new image is to be considered
            explicit or not
        :type is_explicit: bool

        :return: The ID, MD5 sum and final path of the file ingested.
        :rtype: (int, str, str)

        """
        md5 = hashlib.md5(open(image_filepath, 'rb').read()).hexdigest()

        # Attempt to open the file as an image.
        PIL.Image.open(image_filepath)

        file_ext = osp.splitext(image_filepath)[1]

        if self._id_path_map:
            next_uid = max(self._starting_id, max(self._id_path_map) + 1)
        else:
            next_uid = self._starting_id

        # Predict target image file location
        ingest_img_filename = \
            self.INGEST_FILE_TMPL % (md5, next_uid, file_ext)
        ingest_img_filepath = osp.join(self.image_data_dir,
                                       ingest_img_filename)

        # Predict target metadata file location
        ingest_md_filename = \
            self.INGEST_FILE_TMPL % (md5, next_uid, ".bson")
        ingest_md_filepath = osp.join(self.metadata_data_dir,
                                      ingest_md_filename)

        # TODO: Don't copy image/bson file if:
        #       - no paired bson file and the image MD5 already exists in map
        #       - image AND bson MD5s already exist in maps

        # Copying over image file from source location
        shutil.copyfile(image_filepath, ingest_img_filepath)

        # If we were given a metadata file, bring it over and add the md obj
        if metadata_filepath:
            shutil.copyfile(metadata_filepath, ingest_md_filepath)
            self._id_metadata_map[next_uid] = \
                IngestItemMetadata.from_file(ingest_md_filepath)

        self._id_path_map[next_uid] = ingest_img_filepath
        self._md5_id_map.setdefault(md5, []).append(next_uid)

        if is_explicit:
            self.set_explicit(next_uid)

        return next_uid, md5, ingest_img_filepath

    def items(self):
        """
        :return: tuple of (ID, filepath) pairs for files ingested
        :rtype: tuple of (int, str)
        """
        return self._id_path_map.items()

    def iteritems(self):
        """ Return iterator over ingested item (id, filepath) pairs. """
        return self._id_path_map.iteritems()

    def ids(self):
        """
        :return: Tuple of ingested file IDs
        :rtype: tuple of int
        """
        return self._id_path_map.keys()

    def has_id(self, item_id):
        """
        Return whether this ingest contains the given ID or not.

        :param item_id: The item ID to check for
        :type item_id: int

        :return: True if this ingest contains the give ID and False if it does
            not.
        :rtype: bool

        """
        return item_id in self._id_path_map

    def get_img_path(self, item_id):
        """
        Get the file path associated with the given ID, or None if no such ID
        exists in the ingest.

        :param item_id: file id to search for
        :type item_id: int

        :return: Path to the file associated to the given ID or None of no such
            ID exists in our ingest.
        :rtype: str or None

        """
        return self._id_path_map.get(item_id, None)

    def get_metadata(self, item_id):
        """
        Get the associated metadata object for the given image ID. This may
        return None if there is no metadata stored for the given image ID.

        :param item_id: The ID of the image to get the metadata for.
        :type item_id: int

        :return: The image ID's associate metadata object or None if there isn't
            one associated.
        :rtype: IngestItemMetadata or None

        """
        return self._id_metadata_map.get(item_id, None)

    def is_explicit(self, item_id):
        """
        Return whether the given file ID is marked as explicit. If the given ID
        does not refer to an ingested file, False is returned.

        :param item_id: ID of the item to check
        :type item_id: int

        :return: True of the item is marked as explicit and False if not. Also
            false if the given ID doesn't belong to this ingest.
        :rtype: bool

        """
        with self._eid_mod_lock:
            return item_id in self._explicit_ids

    def set_explicit(self, item_id):
        """
        Set the given file ID as explicit. This also updates this ingest's
        explicit ID list file.

        :raises KeyError: The given ID does not belong to this ingest.

        :param item_id: Item ID to set as explicit
        :type item_id: int

        """
        with self._eid_mod_lock:
            # only set and write out if this ID isn't already explicit
            if self.has_id(item_id) and item_id not in self._explicit_ids:
                self._explicit_ids.add(item_id)
                with open(self._explicit_ids_list_file, 'a') as elfile:
                    elfile.write('%d\n' % item_id)
            else:
                raise KeyError(item_id)
