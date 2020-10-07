"""
Utility script to transform a set of descriptors, specified by UUID, with
matching class labels, to a test file usable by libSVM utilities for
train/test experiments.

The input CSV file is assumed to be of the format:

    uuid,label
    ...

This is the same as the format requested for other scripts like
``classifier_model_validation.py``.

This is very useful for searching for -c and -g parameter values for a
training sample of data using the ``tools/grid.py`` script, found in the
libSVM source tree. For example:

    <smqtk_source>/TPL/libsvm-3.1-custom/tools/grid.py \
        -log2c -5,15,2 -log2c 3,-15,-2 -v 5 -out libsvm.grid.out \
        -png libsvm.grid.png -t 0 -w1 3.46713615023 -w2 12.2613240418 \
        output_of_this_script.txt
"""

import csv
import logging
from six.moves import zip

import six

from smqtk.representation import DescriptorSet
from smqtk.utils import (
    cli,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


def default_config():
    return {
        'plugins': {
            'descriptor_set':
                make_default_config(DescriptorSet.get_impls()),
        }
    }


def cli_parser():
    parser = cli.basic_cli_parser(__doc__)

    g_io = parser.add_argument_group("IO Options")
    g_io.add_argument('-f', metavar='PATH',
                      help='Path to the csv file mapping descriptor UUIDs to '
                           'their class label. String labels are transformed '
                           'into integers for libSVM. Integers start at 1 '
                           'and are applied in the order that labels are '
                           'seen in this input file.')
    g_io.add_argument('-o', metavar='PATH',
                      help='Path to the output file to write libSVM labeled '
                           'descriptors to.')
    return parser


def main():
    args = cli_parser().parse_args()
    config = cli.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    #: :type: smqtk.representation.DescriptorSet
    descriptor_set = from_config_dict(
        config['plugins']['descriptor_set'],
        DescriptorSet.get_impls()
    )

    labels_filepath = args.f
    output_filepath = args.o

    # Run through labeled UUIDs in input file, getting the descriptor from the
    # configured index, applying the appropriate integer label and then writing
    # the formatted line out to the output file.
    input_uuid_labels = csv.reader(open(labels_filepath))

    with open(output_filepath, 'w') as ofile:
        label2int = {}
        next_int = 1
        uuids, labels = list(zip(*input_uuid_labels))

        log.info("Scanning input descriptors and labels")
        for i, (la, d) in enumerate(
                    zip(labels, descriptor_set.get_many_descriptors(uuids))):
            log.debug("%d %s", i, d.uuid())
            if la not in label2int:
                label2int[la] = next_int
                next_int += 1
            ofile.write(
                "%d " % label2int[la] +
                ' '.join(["%d:%.12f" % (j+1, f)
                          for j, f in enumerate(d.vector())
                          if f != 0.0]) +
                '\n'
            )

    log.info("Integer label association:")
    for i, la in sorted((i, _la) for _la, i in six.iteritems(label2int)):
        log.info('\t%d :: %s', i, la)


if __name__ == '__main__':
    main()
