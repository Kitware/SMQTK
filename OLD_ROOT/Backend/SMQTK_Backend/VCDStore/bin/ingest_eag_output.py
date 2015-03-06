#!/usr/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Ingest into a VCDStore features for clip IDS given a descriptor label, the clip
ids file, and the corresponding numpy .npy data file containing the features.
The clip IDs file and the numpy data file must be data parallel. i.e. the clip
ID on line N of the clip IDs file must match row N of the numpy data array
from the .npy file.

"""

import numpy as np
import optparse
import os.path as osp

from SMQTK_Backend.VCDStore import VCDStore, VCDStoreElement


### Option parsing
parser = optparse.OptionParser()

required_opts = optparse.OptionGroup(parser, "Required Options")
required_opts.add_option('-d', '--descriptor-name', dest='descriptor_name',
                         help='The name of the descriptor the two files '
                              'represent.')
required_opts.add_option('-c', '--clip-id-file', dest='clip_id_file',
                         help='The path to the clip ID list file.')
required_opts.add_option('-n', '--npy-file', dest='npy_file',
                         help='The path to the numpy data file.')
parser.add_option_group(required_opts)

optional_opts = optparse.OptionGroup(parser, "Optionals")
optional_opts.add_option('--store-file',
                         help='A custom path to the VCDStore file (SQLite3 '
                              'implementation)')
optional_opts.add_option('--overwrite', action='store_true', default=False,
                         help='Overwrite duplicate entries in the VCDStore if '
                              'one already exists at the designated location.')
parser.add_option_group(optional_opts)

opts, args = parser.parse_args()


### Validate / Initialize vars
if not opts.descriptor_name:
    print "ERROR: No descriptor name given. We must have one!"
    exit(1)
descr_name = opts.descriptor_name

if not opts.clip_id_file or not osp.isfile(opts.clip_id_file):
    print "ERROR: Clip ID file not valid (given: '%s')" % opts.clip_id_file
    exit(1)
if not opts.npy_file or not osp.isfile(opts.npy_file):
    print "ERROR: Numpy data file not valid (given: '%s')" % opts.npy_file
    exit(1)

if opts.store_file:
    vcd_store = VCDStore(fs_db_path=opts.store_file)
else:
    vcd_store = VCDStore()

overwrite = opts.overwrite


### Loading up clip IDs and numpy file, pairing parallel rows
print "Extracting clip IDs and features..."
clip_ids = map(int, (l.strip(' \t\n') for l in open(opts.clip_id_file).readlines()))
npy_matrix = np.load(opts.npy_file)
assert len(clip_ids) == npy_matrix.shape[0], \
    "ERROR: Number of clip ids found did not match the numpy matrix size! " \
    "(clip ids: %d) (np arrays: %d)" \
    % (len(clip_ids), npy_matrix.shape[0])
print "-->", len(clip_ids)


### Creating VCDStoreElements
elems = [None] * len(clip_ids)
for i, (cid, np_array) in enumerate(zip(clip_ids, npy_matrix)):
    elems[i] = VCDStoreElement(descr_name, cid, np_array)
print "VCD Elements generated:", len(elems)


### Storage into VCDStore
vcd_store.store_feature(elems, overwrite=overwrite)
