import json
import logging
import logging.handlers
import os
import time


__author__ = "paul.tunison@kitware.com"


def initialize_logging(logger, stream_level=logging.WARNING,
                       output_filepath=None, file_level=None):
    """
    Standard logging initialization.

    :param logger: Logger instance to initialize
    :type logger: logging.Logger

    :param stream_level: Logging level to set for the stderr stream formatter.
    :type stream_level: int

    :param output_filepath: Output logging from the given logger to the provided
        file path. Currently, we log to that file indefinitely, i.e. no
        rollover. Rollover may be added in the future if the need arises.
    :type output_filepath: str

    :param file_level: Logging level to output to the file. This the same as the
        stream level by default.

    """
    log_formatter = logging.Formatter(
        "%(levelname)7s - %(asctime)s - %(name)s.%(funcName)s - %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    if output_filepath:
        # TODO: Setup rotating part of the handler?
        file_handler = logging.handlers.RotatingFileHandler(output_filepath,
                                                            mode='w',
                                                            delay=1)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level or stream_level)
        logger.addHandler(file_handler)

    # Because there are two levels checked before a logging message is emitted:
    #   * the logging object's level
    #   * The stream handlers level
    logger.setLevel(min(stream_level, file_level or stream_level))


def output_config(output_path, config_dict, log=None, overwrite=False,
                  error_rc=1):
    """
    If a valid output configuration path is provided, we output the given
    configuration dictionary as JSON or error if the file already exists (when
    overwrite is False) or cannot be written. We exit the program as long as
    ``output_path`` was given a value, with a return code of 0 if the file was
    written successfully, or the supplied return code (default of 1) if the
    write failed.

    Specified error return code cannot be 0, which is reserved for successful
    operation.

    :raises ValueError: If the given error return code is 0.

    :param output_path: Path to write the configuration file to.
    :type output_path: str

    :param config_dict: Configuration dictionary containing JSON-compliant
        values.
    :type config_dict: dict

    :param overwrite: If we should clobber any existing file at the specified
        path. We exit with the error code if this is false and a file exists at
        ``output_path``.
    :type overwrite: bool

    :param error_rc: Custom integer error return code to use instead of 1.
    ;type error_rc: int

    :param log: Optionally logging instance. Otherwise we use a local one.
    :type log: logging.Logger

    """
    error_rc = int(error_rc)
    if error_rc == 0:
        raise ValueError("Error return code cannot be 0.")
    if log is None:
        log = logging.getLogger(__name__)
    if output_path:
        if os.path.exists(output_path) and not overwrite:
            log.error("Output configuration file path already exists! (%s)",
                      output_path)
            exit(error_rc)
        else:
            log.info("Outputting JSON configuration to: %s", output_path)
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=4, check_circular=True,
                          sort_keys=True)
            exit(0)


def report_progress(log, state, interval):
    """
    Loop progress reporting function that logs (when in debug) loops per
    second, loops in the last reporting period and total loops executed.

    The ``state`` given to this function must be a list of 7 floats, initially
    all set to 0. This function will update the fields of the state as its is
    called to control when reporting should happen and what to report.

    :param log: Logger to send debug message to.
    :type log: logging.Logger

    :param state: Reporting state. This should be initialized to a list of 6
        zeros (floats), and then should not be modified externally from this
        function.
    :type state: list[float]

    :param interval: Frequency in seconds that reporting messages should be
        made. This should be greater than 0.
    :type interval: float

    """
    # State format:
    #   [lc, c, dc, lt, t, dt, st]
    #   [ 0, 1,  2,  3, 4,  5,  6]

    # Starting time
    if not state[6]:
        state[6] = time.time()

    state[1] += 1
    state[4] = time.time()
    state[5] = state[4] - state[3]
    if state[5] >= interval:
        state[2] = state[1] - state[0]
        # TODO: Could possibly to something with ncurses
        #       - to maintain a single
        #       line.
        log.debug("Loops per second %f (avg %f) (%d / %d total)",
                  state[2] / state[5],
                  state[1] / (state[4] - state[6]),
                  state[2], state[1])
        state[3] = state[4]
        state[0] = state[1]
