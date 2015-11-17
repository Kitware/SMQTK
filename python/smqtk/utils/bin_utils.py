import json
import logging
import logging.handlers
import os


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
        # "%(levelname)7s - %(asctime)s - %(name)s.%(funcName)s - %(message)s"
        "%(levelname)7s - %(asctime)s - %(name)s - %(message)s"
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

    :param log: Optionally logging instance. Otherwise we use a local one.
    :type log: logging.Logger

    """

    error_rc = int(error_rc)
    if error_rc == 0:
        raise ValueError("Error return code cannot be 0.")
    if log is None:
        log = logging.getLogger(output_config)
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
