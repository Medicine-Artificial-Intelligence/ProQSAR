import os
import logging
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Iterable, Optional


def _null_handler() -> logging.Handler:
    return logging.NullHandler()


def force_quiet(
    extra_logger_names: Optional[Iterable[str]] = None,
    ignore_warnings: bool = True,
) -> Dict[str, Any]:
    """
    Force-silence Python logging, the `warnings` module, and RDKit logging (if present).

    :param extra_logger_names: Iterable of logger names (strings) to silence in addition to
        all known loggers. Useful for third-party loggers created lazily by name.
    :type extra_logger_names: Optional[Iterable[str]]
    :param ignore_warnings: If True, applies ``warnings.filterwarnings("ignore")``.
    :type ignore_warnings: bool
    :returns: A dictionary capturing previous logger/warnings state which can be passed to
        ``restore_quiet(state)`` to revert the changes.
    :rtype: Dict[str, Any]

    :raises ImportError: If RDKit import fails the function will *not* raise; RDKit silence is optional.
    :example:

    >>> state = force_quiet(["matplotlib", "urllib3"])
    >>> # run noisy code...
    >>> restore_quiet(state)
    """
    state: Dict[str, Any] = {}

    # 1) Record and disable global logging via logging.disable
    state["manager_disable"] = (
        logging.root.manager.disable
    )  # previous global disable level
    logging.disable(logging.CRITICAL)  # block messages <= CRITICAL (practically all)

    # 2) Silence existing loggers in the logging manager
    manager = logging.root.manager
    state["loggers"] = {}
    for name, logger_obj in list(manager.loggerDict.items()):
        try:
            if isinstance(logger_obj, logging.Logger):
                state["loggers"][name] = {
                    "level": logger_obj.level,
                    "handlers": list(logger_obj.handlers),
                    "propagate": logger_obj.propagate,
                }
                # remove handlers and prevent propagation
                logger_obj.handlers = []
                logger_obj.propagate = False
                # set an extremely high level so it won't emit
                logger_obj.setLevel(logging.CRITICAL + 10)
        except Exception:
            # ignore any logger we can't introspect
            pass

    # 3) Silence the root logger (replace handlers with a NullHandler)
    root = logging.getLogger()
    state["root"] = {"level": root.level, "handlers": list(root.handlers)}
    try:
        root.handlers = []
        root.addHandler(_null_handler())
        root.setLevel(logging.CRITICAL + 10)
    except Exception:
        # best-effort only
        pass

    # 4) Optionally silence warnings module
    if ignore_warnings:
        state["warnings_filters"] = warnings.filters[:]  # copy current filters
        warnings.filterwarnings("ignore")

    # 5) Try to disable RDKit RDLogger (if available)
    state["rdkit_disabled"] = False
    try:
        from rdkit import RDLogger  # type: ignore

        RDLogger.DisableLog("rdApp.*")
        state["rdkit_disabled"] = True
    except Exception:
        # RDKit not installed or import failed: ignore
        state["rdkit_disabled"] = False

    # 6) Optionally silence specific third-party logger names passed explicitly
    if extra_logger_names:
        state["extra_loggers"] = {}
        for name in extra_logger_names:
            lg = logging.getLogger(name)
            state["extra_loggers"][name] = {
                "level": lg.level,
                "handlers": list(lg.handlers),
                "propagate": lg.propagate,
            }
            lg.handlers = []
            lg.propagate = False
            lg.setLevel(logging.CRITICAL + 10)

    return state


def _restore_manager_disable(state: Dict[str, Any]) -> None:
    try:
        prev_disable = state.get("manager_disable", 0)
        logging.root.manager.disable = prev_disable
    except Exception:
        # best-effort: ignore failures here
        pass


def _restore_loggers(state: Dict[str, Any]) -> None:
    loggers_state = state.get("loggers", {})
    if not isinstance(loggers_state, dict):
        return

    for name, info in loggers_state.items():
        try:
            lg = logging.getLogger(name)
            lg.handlers = info.get("handlers", [])
            lg.propagate = info.get("propagate", True)
            lg.setLevel(info.get("level", logging.NOTSET))
        except Exception:
            # ignore individual logger restore failures
            continue


def _restore_root(state: Dict[str, Any]) -> None:
    root_state = state.get("root")
    if not root_state:
        return
    try:
        root = logging.getLogger()
        root.handlers = root_state.get("handlers", [])
        root.setLevel(root_state.get("level", logging.NOTSET))
    except Exception:
        pass


def _restore_warnings(state: Dict[str, Any]) -> None:
    if "warnings_filters" in state:
        try:
            warnings.filters[:] = state["warnings_filters"]
        except Exception:
            try:
                warnings.resetwarnings()
            except Exception:
                pass


def _restore_rdkit(state: Dict[str, Any]) -> None:
    if not state.get("rdkit_disabled"):
        return
    try:
        # try the standard re-enable API first
        from rdkit import RDLogger  # type: ignore

        try:
            # newer RDKit versions may offer EnableLog
            RDLogger.EnableLog("rdApp.*")
            return
        except Exception:
            # fallback: try adjusting the RDKit logger level
            try:
                rl = RDLogger.logger()
                rl.setLevel(logging.NOTSET)
            except Exception:
                pass
    except Exception:
        # RDKit isn't available — nothing to restore
        pass


def _restore_extra_loggers(state: Dict[str, Any]) -> None:
    extra = state.get("extra_loggers", {})
    if not isinstance(extra, dict):
        return

    for name, info in extra.items():
        try:
            lg = logging.getLogger(name)
            lg.handlers = info.get("handlers", [])
            lg.propagate = info.get("propagate", True)
            lg.setLevel(info.get("level", logging.NOTSET))
        except Exception:
            continue


# --- public restore_quiet ----------------------------------------------------


def restore_quiet(state: Dict[str, Any]) -> None:
    """
    Restore the logging/warnings/RDKit state previously returned by `force_quiet`.

    :param state: The dictionary returned by `force_quiet`.
    :type state: Dict[str, Any]
    :returns: None
    :rtype: None
    """
    if not state:
        return

    # Call small helpers; each helper is responsible for try/except around its own work.
    _restore_manager_disable(state)
    _restore_loggers(state)
    _restore_root(state)
    _restore_warnings(state)
    _restore_rdkit(state)
    _restore_extra_loggers(state)


@contextmanager
def quiet(
    extra_logger_names: Optional[Iterable[str]] = None, ignore_warnings: bool = True
):
    """
    Context manager that calls `force_quiet()` on enter and `restore_quiet()` on exit.

    :param extra_logger_names: Optional iterable of additional logger names to silence.
    :type extra_logger_names: Optional[Iterable[str]]
    :param ignore_warnings: If True, ignore warnings inside the context.
    :type ignore_warnings: bool
    :returns: Yields None (context manager).
    :rtype: None

    :example:

    >>> with quiet(["matplotlib", "urllib3"]):
    ...     # noisy tasks run silently here
    ...     do_noisy_work()
    """
    state = force_quiet(
        extra_logger_names=extra_logger_names, ignore_warnings=ignore_warnings
    )
    try:
        yield
    finally:
        restore_quiet(state)


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
