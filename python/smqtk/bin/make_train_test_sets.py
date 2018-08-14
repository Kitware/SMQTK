#!/usr/bin/env python
import argparse
import csv
import itertools
import os
import re

import numpy
import six
from sklearn.model_selection import StratifiedShuffleSplit


class KeyToFilepath(argparse.Action):
    """
    Custom argparse action for parsing out positional class-to-filepath
    arguments.
    """
    re_key2path = re.compile('(\w+)=(.+)', flags=re.UNICODE)

    # noinspection PyUnusedLocal,PyShadowingBuiltins
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        """
        Custom constructor to enforce that `nargs` is always `+`.
        """
        super(KeyToFilepath, self).__init__(option_strings,
                                            dest, "+",
                                            const, default, type,
                                            choices, required,
                                            help, metavar)

    # noinspection PyShadowingNames
    def __call__(self, parser, namespace, values, option_string=None):
        d = dict()
        for a in values:
            m = self.re_key2path.match(a)
            if not m:
                raise ValueError("Invalid argument syntax: '%s'" % a)
            cls_name = m.group(1)
            filepath = m.group(2)
            if not os.path.isfile(filepath):
                raise ValueError(
                    "Invalid filepath '%s' given for argument: '%s'"
                    % (filepath, a)
                )
            # Read in UIDs from lines in CSV file
            d[cls_name] = filepath
        setattr(namespace, self.dest, d)


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('cls_to_cmdProcessedCsv',
                        nargs='+',
                        help="Series of `label=filepath` arguments where we "
                             "interpret the string value before the `=` sign "
                             "as the class label and the value after to be the "
                             "path to the `compute_many_descriptors` output "
                             "CSV of files successfully processed.",
                        action=KeyToFilepath)
    parser.add_argument('-t', '--test-percent',
                        type=float,
                        default=0.3,
                        help="Percentage of images per class to split for "
                             "testing. Should be in the [0,1] range. Selects "
                             "~30%% by default.")
    parser.add_argument('--rand-state', type=int, default=0,
                        help='Random state initialization integer. Default is '
                             '0.')
    parser.add_argument('--output-base',
                        default="classifier",
                        help="String base to output files. We will generate 3 "
                             "files: '<>.all_uuids.csv', '<>.train_uuids.csv' "
                             "and '<>.test_uuids.csv'. "
                             "Default is 'classifier'.")
    return parser


def main():
    args = cli_parser().parse_args()

    TEST_PERCENT = args.test_percent
    RAND_STATE = args.rand_state
    OUTPUT_BASE = args.output_base
    CLS_TO_FILEPATH = args.cls_to_cmdProcessedCsv

    # Parse CSV files associated to classes
    cls_uuids = {}
    for cls, filepath in six.iteritems(CLS_TO_FILEPATH):
        cls_uuids[cls] = sorted({r[1] for r in csv.reader(open(filepath))})

    cls_list = sorted(cls_uuids)
    all_label, all_uuids = \
        zip(*[(cls_name, uuid)
              for cls_name in cls_list
              for uuid in cls_uuids[cls_name]])
    # Transform into numpy array for multi-index access later
    all_label = numpy.array(all_label)
    all_uuids = numpy.array(all_uuids)

    # ``n_splits=1``  -- Only make one train/test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_PERCENT,
                                 random_state=RAND_STATE)

    # Get array of index position values of ``all_uuids`` of uuids to use for
    # train and test sets, respectively.
    train_index, test_index = \
        iter(sss.split(numpy.zeros(len(all_label)), all_label)).next()
    uuids_train, uuids_test = all_uuids[train_index], all_uuids[test_index]
    label_train, label_test = all_label[train_index], all_label[test_index]

    print("Train:")
    for cls_label in cls_list:
        cnt = label_train.tolist().count(cls_label)
        print("- %s:\t%d\t(~%.2f %% of total class examples)"
              % (cls_label, cnt, float(cnt) / len(cls_uuids[cls_label]) * 100))
    print("Test:")
    for cls_label in cls_list:
        cnt = label_test.tolist().count(cls_label)
        print("- %s:\t%d\t(~%.2f %% of total class examples)"
              % (cls_label, cnt, float(cnt) / len(cls_uuids[cls_label]) * 100))

    # Save out files for use with ``classifier_model_validation``
    with open('%s.all_uuids.csv' % OUTPUT_BASE, 'w') as f:
        w = csv.writer(f)
        for uuid, label in itertools.izip(all_uuids, all_label):
            w.writerow([uuid, label])

    with open('%s.train_uuids.csv' % OUTPUT_BASE, 'w') as f:
        w = csv.writer(f)
        for uuid, label in itertools.izip(uuids_train, label_train):
            w.writerow([uuid, label])

    with open('%s.test_uuids.csv' % OUTPUT_BASE, 'w') as f:
        w = csv.writer(f)
        for uuid, label in itertools.izip(uuids_test, label_test):
            w.writerow([uuid, label])


if __name__ == '__main__':
    main()
