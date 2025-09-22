import os
import logging


def setup_logging(log_level: str = "INFO", log_filename: str = None) -> logging.Logger:
    """
    Configure and return a root logger that writes to console or a file.

    This helper resets existing handlers on the root logger and configures
    a simple formatter. If `log_filename` is provided the logger will append
    to that file (creating parent directories if necessary); otherwise it
    logs to the console (stderr) via the default StreamHandler.

    :param log_level: Logging level to set. Case-insensitive. Typical values:
                      ``'DEBUG'``, ``'INFO'``, ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``.
    :type log_level: str
    :param log_filename: Optional path to a file where logs will be appended.
                         If ``None`` (default) logs are emitted to the console.
    :type log_filename: str or None

    :return: Configured root logger instance.
    :rtype: logging.Logger

    :raises ValueError: If `log_level` is not a valid logging level name.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger()
    logger.handlers.clear()  # Efficiently remove all existing handlers

    if log_filename:
        # Ensure target directory exists before configuring file handler
        parent_dir = os.path.dirname(log_filename)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        logging.basicConfig(
            level=numeric_level, format=log_format, filename=log_filename, filemode="a"
        )
    else:
        logging.basicConfig(level=numeric_level, format=log_format)

    return logger
