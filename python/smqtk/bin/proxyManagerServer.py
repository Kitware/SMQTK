"""
Server for hosting proxy manager which hosts proxy object instances.

This takes a simple configuration file that looks like the following:

|   [server]
|   port = <integer>
|   authkey = <string>

"""
# Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
# LICENSE.txt for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

from smqtk.utils import bin_utils
from smqtk.utils.proxy_manager import ProxyManager


def default_config():
    return {
        "port": 5000,
        "authkey": "CHANGE_ME",
    }


def cli_parser():
    return bin_utils.basic_cli_parser(__doc__)


def main():
    parser = cli_parser()
    args = parser.parse_args()

    # Default config options for this util are technically valid for running,
    # its just a bad authkey.
    config = bin_utils.utility_main_helper(default_config, args,
                                           default_config_valid=True)

    port = int(config['port'])
    authkey = str(config['authkey'])

    mgr = ProxyManager(('', port), authkey)
    mgr.get_server().serve_forever()


if __name__ == '__main__':
    main()
