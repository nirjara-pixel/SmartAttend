"""
SmartAttend - Application Logging Module
Configures thread-safe rotating file logging to logs/app.log and console output,
and installs global exception handlers for main and worker threads.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import threading
from typing import Optional

import config


_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "SmartAttend") -> logging.Logger:
    """
    Get or initialize the shared application logger.
    Logs to logs/app.log and console.
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    # Ensure logs directory exists
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOG_FILE

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if reloaded
    if not logger.handlers:
        # 1. Rotating File Handler (up to 5 MB per file, 3 backups)
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)-8s] [%(threadName)-12s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 2. Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _LOGGER = logger
    return _LOGGER


def install_global_exception_hooks(logger: Optional[logging.Logger] = None) -> None:
    """
    Install hooks to capture any unhandled exceptions in the main thread
    or background worker threads and record them in logs/app.log.
    """
    log = logger or get_logger()

    def main_thread_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log.critical(
            "Unhandled exception on main thread:",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    def thread_excepthook(args: threading.ExceptHookArgs):
        if issubclass(args.exc_type, KeyboardInterrupt):
            return
        log.critical(
            f"Unhandled exception in background thread '{args.thread.name}':",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )

    sys.excepthook = main_thread_excepthook
    threading.excepthook = thread_excepthook
