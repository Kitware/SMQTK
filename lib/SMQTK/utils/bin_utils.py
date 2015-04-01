
import logging
import logging.handlers
import optparse


class SMQTKOptParser (optparse.OptionParser):
    """
    Class override to change formatting for description and epilogue strings.
    """

    def format_description(self, formatter):
        return self.description or ''

    def format_epilog(self, formatter):
        return self.epilog or ''


def initializeLogging(logger, stream_level=logging.WARNING,
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
