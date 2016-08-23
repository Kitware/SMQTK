"""
Add a set of local system files to a data set via explicit paths or shell-style
glob strings.
"""

import glob
import json
import logging
import os.path as osp

from smqtk.representation import get_data_set_impls
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import bin_utils, plugin


def default_config():
    return {
        "data_set": plugin.make_config(get_data_set_impls())
    }


def cli_parser():
    parser = bin_utils.basic_cli_parser(__doc__)
    parser.add_argument("input_files", metavar='GLOB', nargs='*')
    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()
    config = bin_utils.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    log.debug("Script arguments:\n%s" % args)

    def iter_input_elements():
        for f in args.input_files:
            f = osp.expanduser(f)
            if osp.isfile(f):
                yield DataFileElement(f)
            else:
                log.debug("Expanding glob: %s" % f)
                for g in glob.glob(f):
                    yield DataFileElement(g)

    log.info("Adding elements to data set")
    #: :type: smqtk.representation.DataSet
    ds = plugin.from_plugin_config(config['data_set'], get_data_set_impls())
    ds.add_data(*iter_input_elements())


if __name__ == '__main__':
    main()
