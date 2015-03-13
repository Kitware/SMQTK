#!/bin/env python
"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


Program to distribute image processing load over a cluster.

Inputs:
    - location of worklist_file
    - location of output directory

Operational narative:

The rank(0) node manages the work list as a server. The other nodes
are clients and request work from the server.

The server opens the worklist file and starts a receive operation.
Once the receive completes, the server handles the request. Requests are
simple text strings defined as follows:
    'feed-me'
        - client wants a work item
    'compete path1 [path2 ...]'
        - client is done with work item and has generated the files specified by
          the proceeding, space separated elements. These files should be
          pickled VCDStoreElement objects.

Messages sent to clients:
    'proc-file <filename>'
        - client is to process specified file, generating one or more pickled
          VCDStoreElement objects (one per file?) for storage by the server.
          This is communicated by sending a 'complete ...' command back to the
          server.
    'die-sucka'
        - indicates no more work, client is to terminate.

This scheduler generates a set of lig files in the specified directory.
The log files are timestamped and named with the  process RANK so the
files will sort in groups based on the time the scheduler started.
There may be some skewing in the timestamp because each scheduler instance
gets its own time.


Descriptor Client Methods
-------------------------
Each descriptor mode should supply a .py file in the "descriptor_modules"
directory that defines two methods with the following names and signatures:

    generate_config( config=None:(SafeConfigCommentParser or None) )
        -> returns SafeConfigCommentParser
        -> Defines options needed by descriptor for user to fill in.

    process_file( config:SafeConfigCommentParser, working_dir:str,
                  image_dir:str, video_file:str )
        -> returns iterable of VCDFeatureStoreElements, ideally being clip lever
           features.
        -> Defines how the descriptor runs (ideally with a VCDWorker, but we
           don't want client workers to deal with the VCDStore).
        -> Returns a iterable, preferably a tuple or list, of produced
           VCDStoreElements.

Each file should start with an lower-case or capital letter, as anything else
will be ignored.

"""

import _abcoll
import cPickle
import logging
from mpi4py import MPI
import os
import os.path as osp
import optparse
import sys
import traceback
import time

from SMQTK_Backend.utils import SafeConfigCommentParser
from SMQTK_Backend.VCDWorkers import VCDWorkerInterface
from SMQTK_Backend.VCDStore import VCDStore, errors as VCDErrors

from SMQTK_Backend.VCDWorkers.descriptor_modules import \
    get_descriptor_modules

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()

vc_calls = 0
vc_hits = 0
file_count = 0
start_time = 0.0
end_time = 0.0
cached_output = 0


MPIS_CONFIG_SECT = 'vcd_scheduler'


# ================================================================
def parse_options():
    """
    Parse command line options.

    The command line options are defined here and then parsed.
    The resulting namespace object is returned.

    :rtype: (optparse.Values, list of str)

    """
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)

    parser.add_option('-c', '--config', dest='config_files',
                      action='append', default=[],
                      help="Configuration file for the mpi scheduler. "
                           "If the config file is in a different location, "
                           "the path should be specified with this option. "
                           "Multiple config files may be provided, allowing "
                           "different files to contain partial configurations "
                           "that are combined by this script. If multiple "
                           "config specify the same section/option pair, the "
                           "value taken will be from last config file "
                           "containing that option. Thus, it is a good idea "
                           "to specify general partial configs first, followed "
                           "by more specific config files.")

    opt_group = optparse.OptionGroup(parser, "Optionals")
    opt_group.add_option('--dry-run', action='store_true', default=False,
                         help='do not process video on client side')
    opt_group.add_option('-o', '--output-config', type=str,
                         help='Output a configuration file to the given file '
                              'path.')
    opt_group.add_option('--overwrite', action='store_true', default=False,
                         help='Allow overwriting of an existing file when '
                              'outputting a configuration file.')

    v_group = optparse.OptionGroup(parser, "Verbosity/Logging")
    v_group.add_option('-v', action='count', default=0, dest='verbosity',
                       help='Make output more verbose. By default, only '
                            'informational messages are shown. Additional '
                            'verbosity will show debugging messages')

    parser.add_option_group(opt_group)
    parser.add_option_group(v_group)

    return parser.parse_args()


# ================================================================
def generate_config(config=None):
    """
    Generate a sample configuration object.

    :param config: An optional configuration object to add to.
    :type config: None or SafeConfigCommentParser

    :return: Generated configuration object
    :rtype: SafeConfigCommentParser

    """
    log = logging.getLogger()

    if config is None:
        log.debug('generating new config object')
        config = SafeConfigCommentParser()

    log.info("Adding vcd_scheduler section...")
    sect = MPIS_CONFIG_SECT
    config.add_section(sect)
    config.set(sect, 'image_directory', '# REQUIRED',
               "The root directory where imagery will be extracted and stored. "
               "This is a common structure across any descriptors that want to "
               "extract imagery.")
    config.set(sect, 'run_work_directory', '# REQUIRED',
               "The base working directory for this run.")
    config.set(sect, 'video_work_list', '# REQUIRED',
               "A test time with a new-line separated list of video files to "
               "process.")
    config.set(sect, 'log_dir', "%(run_work_directory)s/logs",
               "Directory where log files from the run are to be stored.")
    config.set(sect, 'target_vcd_store', 'SQLiteVCDStore.db',
               "The sqlite3 database file results will be accumulated into. If "
               "this is given as a relative path it will be considered "
               "relative to the provided work directory.")
    config.set(sect, 'vcd_store_overwrite', 'false',
               "When the server node attempts to store VCDStoreElement "
               "objects, overwrite existing entries if found. Else, if this is "
               "false, favor the existing entries and roll back the insertion "
               "attempt.")

    # TODO: [generate_config] (see below)
    # This tells the vcd_scheduler which descriptor client module to load
    # This should also tell us what VCDWorker implementation to use. This
    #   requires that we shift over to using the VCDWorker implementation
    #   structure instead of the very similar function thing we currently have
    #   going on...
    # Should be straight forward to convert current stuff to VCDWorker impl.
    #   Need to modify VCDWorker impls with lessons learned here.
    # avail_modes = get_descriptor_client_methods().keys()
    descr_modules = get_descriptor_modules()
    config.set(sect, 'descriptor_mode', '# REQUIRED',
               "The descriptor mode to run in. This defined which descriptor "
               "will process the videos listed in the given work list file.\n"
               "\n"
               "The available modes are as follows:\n"
               " - '%s'"
               % "'\n - '".join(descr_modules.keys()))

    # Append descriptor module configs to this one
    for name, m in descr_modules.items():
        log.info("Adding '%s' module section...", name)
        m.generate_config(config)

    return config


# ================================================================
log_print = logging.getLogger('vcd_scheduler').info
log_debug = logging.getLogger('vcd_scheduler').debug
log_warn = logging.getLogger('vcd_scheduler').warn
log_error = logging.getLogger('vcd_scheduler').error


# ================================================================
def print_server_stats():
    global file_count
    global start_time
    global end_time

    log_print("*****************************************")
    log_print("Total video files processed: %i" % file_count)
    log_print("Elapsed time (sec): %f" % (end_time - start_time))
    if file_count == 0:
        rate = 0
    else:
        rate = (end_time - start_time) / file_count
    log_print("File processing rate(s/f): %f" % rate)


# ================================================================
def print_client_stats():
    global vc_hits
    global vc_calls
    global file_count
    global start_time
    global end_time
    global cached_output

    log_print("*****************************************")
    log_print( "Client statistics for node")
    log_print( "Total video files processed: %i" % file_count)
    log_print( "Elapsed time (sec): %f" % (end_time - start_time))
    if file_count == 0:
        rate = 0
    else:
        rate = (end_time - start_time) / file_count
    log_print("File processing rate(s/f): %f" % rate)


# ================================================================
def get_vcd_element_dir(base_work_dir):
    """
    get standard path to VCDStoreElement serialization storage space for a given
    work directory

    -> ".../store_elements/"

    """
    return osp.join(base_work_dir, 'store_elements')


# ================================================================
def get_vcd_element_file(descr_mode, video_file, work_dir):
    """
    Get standard path to VCDStoreElement serialized file for a given
    work directory, descriptor mode and video.

    If the containing directory to this file path did not exist, It will be
    created by return.

    -> ".../store_elements/<vpfx>/<vkey>/<dscr>.<key>.elements"
    where "<...>" is filled in with relevant material for given video and
    descriptor.

    """
    prefix, key = VCDWorkerInterface.get_video_prefix(video_file)
    d = osp.join(get_vcd_element_dir(work_dir), descr_mode, prefix, key)
    d = VCDWorkerInterface.create_dir(d)
    f = '%s.%s.elements' % (descr_mode, key)
    return osp.join(d, f)


# ================================================================
def server_side(config):
    """
    Server implementation
    """
    global file_count

    # count of outstanding work items
    eof_seen = False
    terminate_count = mpi_size-1

    log_print("I'll be your server today on %s with a party of %i"
              % (os.getcwd(), mpi_size))

    work_list = config.get(MPIS_CONFIG_SECT, 'video_work_list')
    fd = open(work_list, 'r')  # open work list

    # The VCDStore in which we put deserialized elements.
    work_dir = config.get(MPIS_CONFIG_SECT, 'run_work_directory')
    target_vcds = config.get(MPIS_CONFIG_SECT, 'target_vcd_store')
    vcd_store_overwrite = config.getboolean(MPIS_CONFIG_SECT,
                                            'vcd_store_overwrite')
    vcd_store = VCDStore(db_root=work_dir, fs_db_path=target_vcds)

    descr_mode = config.get(MPIS_CONFIG_SECT, 'descriptor_mode')

    while not (eof_seen and terminate_count <= 0):
        log_print("Waiting for work request")
        req_status = MPI.Status()
        request = comm.recv(source=MPI.ANY_SOURCE, status=req_status)
        log_print("Server request: [%s] from %i"
                  % (request, req_status.source))

        request = request.split()
        command = request[0]
        args = request[1:]

        if command == "feed-me":  # --- command ---
            worker_fed = False
            while not worker_fed:
                #: :type: str
                work_item = fd.readline()
                if work_item:
                    work_item = work_item.strip()  # strip trailing newline
                    # If the predicted result file already exists, then
                    # processing must have already occurred. Don't bother
                    # calling the module process function if this is the case.
                    element_file = get_vcd_element_file(descr_mode, work_item,
                                                        work_dir)
                    if osp.isfile(element_file) and not vcd_store_overwrite:
                        log_print("Existing element file discovered for video "
                                  "'%s', skipping assumed redundant processing "
                                  "for video." % work_item)
                        # no messages sent, cycle to next work item / EoF
                    else:
                        log_print("Sending work element %d, file %s to node %i"
                                  % (file_count, work_item, req_status.source))
                        comm.send("proc-file " + work_item,
                                  dest=req_status.source)
                        file_count += 1
                        worker_fed = True
                else:
                    eof_seen = True
                    comm.send('die-sucka', dest=req_status.source)
                    log_print("*** sending terminate to %i" % req_status.source)
                    terminate_count -= 1
                    log_print("Terminate count: %i" % terminate_count)
                    worker_fed = True

        elif command == "complete":  # --- command ---
            log_print("Worker %d completed a work item!" % req_status.source)

            # Existence check of processed file handled before sending work to
            # client, so just de-serialize and store whatever the client sends
            # back after being told to process something (if anything is even
            # sent back besides completion message)
            if args:  # may be empty I guess? if client sends back nothing
                serialized_file = args[0]  # expecting only one other argument
                with open(serialized_file) as f:
                    elements = cPickle.load(f)
                log_print("Attempting storage of elements")
                for e in elements:
                    log_debug('-> %s', str(e))
                try:
                    vcd_store.store_feature(elements, vcd_store_overwrite)
                    log_print("Storage successful")
                except VCDErrors.VCDDuplicateFeatureError, ex:
                    log_warn("Attempted duplicate element insertion, "
                             "rolling back element insertion for dump -> %s\n"
                             "|--> (error: %s)"
                             % (serialized_file, str(ex)))
            else:
                log_print("No serialized objects returned from client, nothing "
                          "to do.")

        else:
            log_print("Unrecognised command from node %d" % req_status.source)

    fd.close()  # close work list
    log_print("**** Server side terminating")


# ================================================================
def client_side(config, dry_run=False):
    """
    Client side implementation
    """
    global file_count

    log_print("Client starting")

    # Get pertinent configuration values
    work_dir = config.get(MPIS_CONFIG_SECT, 'run_work_directory')
    image_dir = config.get(MPIS_CONFIG_SECT, 'image_directory')
    descr_mode = config.get(MPIS_CONFIG_SECT, 'descriptor_mode')

    # get configured module
    descr_module = get_descriptor_modules()[descr_mode]
    descriptor = descr_module(config, work_dir, image_dir)
    log_print("module name: %s" % descr_module.__name__)

    while True:
        log_print("sending request for work")
        comm.send('feed-me', dest=0)
        reply = comm.recv(source=0)
        task_orders = reply.strip().split(' ')

        # end of input
        if task_orders[0] == 'die-sucka':
            log_print("Received terminate command")
            break

        # video file to process
        elif task_orders[0] == 'proc-file':
            video_file = task_orders[1]
            if not dry_run:
                log_print("generating store elements via descriptor")
                vcd_store_elems = descriptor.process_video(video_file)

                # if any elements were returned at all...
                if vcd_store_elems:
                    assert isinstance(vcd_store_elems, _abcoll.Iterable), \
                        "Module process_file function did not return an " \
                        "iterable!"

                    # Store elements to file in a sub-location within the work
                    # directory, sending the paths to those files back to the
                    # server node.
                    dump_file = get_vcd_element_file(descr_mode, video_file,
                                                     work_dir)
                    log_print("dumping serialized store elements -> %s"
                              % dump_file)
                    with open(dump_file, 'w') as f:
                        cPickle.dump(tuple(vcd_store_elems), f,
                                     cPickle.HIGHEST_PROTOCOL)

                    # Send back to the server node the serialized files for
                    # storage
                    log_print("Sending serialized file path to server")
                    comm.send('complete %s' % dump_file, dest=0)

            else:
                log_print("Dry run on video: %s" % video_file)
            file_count += 1

        # add other commands here

    log_print("**** Client side exiting")
    print_client_stats()


# ================================================================
# Main entry
# - Start the program
#
# TODO: Functionalize this and move this implementation into an importable
#       location, creating a wrapper script for bin.
#       - Will probably want to just use this as the VCD run system within the
#         SmqtkController/VCDSController instead of making two discrete run
#         subsystems.
if __name__ == '__main__':
    # process options
    opts, _ = parse_options()

    # initially load config files if any given (potentially for output config)
    config = SafeConfigCommentParser()
    config.read(opts.config_files)

    ###
    # Config file output if requested
    #
    if opts.output_config:
        logging.basicConfig()
        log = logging.getLogger()
        log.setLevel(logging.WARNING - (10 * opts.verbosity))

        log.info("Outputting configuration file@ %s", opts.output_config)
        if osp.exists(opts.output_config) and not opts.overwrite:
            log.error("Target already exists. Not overwriting.")
            exit(2)

        log.info("Generating config object")
        default_config = generate_config()

        default_config.update(config)

        log.info("Writing config file")
        with open(opts.output_config, 'w') as output_file:
            default_config.write(output_file)
            log.info("Done")
        log.info("Exiting")
        exit(0)

    ###
    # We need at least one config file by this point, so check that that is so
    #
    if not opts.config_files:
        print "ERROR - no configuration file(s) given. (no '-c')"
        exit(1)
    else:
        # Check if files even existed since config read will not error if a file
        # given didn't actually exist. But, I believe the user should know if
        # they're trying to supply a config file that doesn't actually exist...
        for cf in opts.config_files:
            if not osp.isfile(cf):
                print "ERROR - Configuration file doesn't exist."
                print "ERROR - Given:", opts.config_files
                print "ERROR - Doesn't exist:", cf
                exit(1)


    # Taking out some variables we want here in the main function
    pnorm = lambda p: osp.abspath(osp.expanduser(p))
    img_dir = pnorm(config.get(MPIS_CONFIG_SECT, 'image_directory'))
    work_dir = pnorm(config.get(MPIS_CONFIG_SECT, 'run_work_directory'))
    log_dir = pnorm(config.get(MPIS_CONFIG_SECT, 'log_dir'))
    work_list_file = pnorm(config.get(MPIS_CONFIG_SECT, 'video_work_list'))

    descr_module = config.get(MPIS_CONFIG_SECT, 'descriptor_mode')

    # obviously need a work file
    if not (work_list_file and osp.isfile(work_list_file)):
        print "ERROR - Must specify work-list file in config"
        print "ERROR - Given: %s" % work_list_file
        sys.exit(1)

    ###
    # Making sure required directories exist.
    #
    # Due to parallelism, only the rank 0 node will create them. All other ranks
    # will wait until they exist before proceeding as they may want to use them
    # before the rank 0 node gets to creating them.
    #
    dirs_we_care_about = [
        work_dir, log_dir, img_dir,
        get_vcd_element_dir(work_dir)
    ]
    if rank:
        # client node: wait till directories exist
        while not all([osp.isdir(d) for d in dirs_we_care_about]):
            time.sleep(0.1)
    else:
        # server node: create directories if they don't exist yet
        try:
            for d in dirs_we_care_about:
                VCDWorkerInterface.create_dir(d)
        except Exception, ex:
            print "ERROR - Exception occurred : %s" % str(ex)
            print "ERROR - Exception traceback:\n%s" % traceback.format_exc()
            comm.Abort()
            raise

    ###
    # Basic logging setup
    #
    log_fmt_str = '%(levelname)7s - %(asctime)s - %(name)s.%(funcName)s - ' \
                  '%(message)s'
    log_level = logging.WARNING - (10 * opts.verbosity)
    root_log = logging.getLogger()
    log_file_name = osp.join(log_dir,
                             "%s-vcd_scheduler-%04i.log"
                             % (time.strftime("%Y_%j_%H%M%S"), rank))

    log_formatter = logging.Formatter(fmt=log_fmt_str)
    file_log_handler = logging.FileHandler(log_file_name, mode='w')
    file_log_handler.setFormatter(log_formatter)
    file_log_handler.setLevel(log_level)

    root_log.setLevel(1)  # allows handler to control logging level
    root_log.addHandler(file_log_handler)

    ###
    # Start appropriate runtime for rank
    #
    try:
        log_print("Process Starting")

        start_time = time.time()
        if rank == 0:
            # server implementation
            server_side(config)
            stats_func = print_server_stats

        else:
            client_side(config, opts.dry_run)
            stats_func = print_client_stats

        end_time = time.time()
        stats_func()

        log_print("Process Ending")

    # if anything bad goes wrong, at least close comm and re-raise
    except Exception, ex:
        root_log.critical("Exception occurred : %s", str(ex))
        root_log.critical("Exception traceback:\n%s", traceback.format_exc())
        comm.Abort()
        raise
