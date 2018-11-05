"""
Add a set of local system files to a data set via explicit paths or shell-style
glob strings.
"""

import glob
import logging
import os.path as osp

from smqtk.representation import DataSet
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import cli
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)


def default_config():
    return {
        "data_set": make_default_config(DataSet.get_impls())
    }


def cli_parser():
    parser = cli.basic_cli_parser(__doc__)
    parser.add_argument("input_files", metavar='GLOB', nargs='*')
    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()
    config = cli.utility_main_helper(default_config, args)
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
    ds = from_config_dict(config['data_set'], DataSet.get_impls())
    ds.add_data(*iter_input_elements())


if __name__ == '__main__':
    main()
