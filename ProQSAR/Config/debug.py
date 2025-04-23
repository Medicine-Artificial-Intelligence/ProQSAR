import os
import logging


def setup_logging(log_level: str = "INFO", log_filename: str = None) -> logging.Logger:
    """
    Configures logging to either the console or a file based on provided parameters.

    Parameters
    ----------
    log_level : str, optional
        Logging level to set. Defaults to 'INFO'. Options include 'DEBUG', 'INFO',
        'WARNING', 'ERROR', 'CRITICAL'.
    log_filename : str, optional
        If provided, logs are written to this file. Defaults to None,
        which logs to console.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Raises
    ------
    ValueError
        If an invalid log level is provided.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger()
    logger.handlers.clear()  # Efficiently remove all existing handlers

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            level=numeric_level, format=log_format, filename=log_filename, filemode="a"
        )
    else:
        logging.basicConfig(level=numeric_level, format=log_format)

    return logger
