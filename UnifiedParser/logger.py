# Copyright © 2022 Arm China Co. Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# cython: language_level=3
# !/usr/bin/python
# -*- coding: UTF-8 -*-

# !/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
import sys
import os
import traceback

__all__ = [
    "LOGGER",
    "INFO",
    "WARN",
    "DEBUG",
    "ERROR",
    "FATAL",
    "get_error_count",
    "set_log_file"
]

LOGFILE = None
ERRFILE = None


def _logger_stderr_write(s):
    if ERRFILE is not None:
        ERRFILE.write(s)
    sys.stderr.write(s)


def _logger_stdout_write(s):
    if LOGFILE is not None:
        LOGFILE.write(s)
    sys.stdout.write(s)


def _set_logger_file(logfile, errfile):
    global ERRFILE
    global LOGFILE
    LOGFILE = open(logfile, "w", 1)
    if errfile is None or len(errfile) == 0 or errfile == logfile:
        ERRFILE = LOGFILE
    else:
        ERRFILE = open(errfile, "w", 1)


DEFAULT_LEVEL = 2
_ERROR_LEVEL = 9
_INFO_LEVEL = 5
_WARN_LEVEL = 2
_DEBUG_LEVEL = 1
LOGLEVEL = os.environ.get("AIPUBUILDER_LOG", DEFAULT_LEVEL)
LOGLEVEL = int(LOGLEVEL) if isinstance(LOGLEVEL, str) else LOGLEVEL
ERROR_COUNT = 0


class AIPULogger(object):
    def __init__(self, base=""):
        self.base = base
        self.err_count = 0
        self.warning_count = 0
        self._init_msg_header()
        self._hat = "^"
        if os.name != 'nt' and sys.stdout.isatty() and sys.stderr.isatty():
            self._err_msg = '\x1B[31;1m[E]\x1B[0m '
            self._warn_msg = "\x1B[34;1m[W]\x1B[0m "
            self._db_msg = '\x1B[33;1m[D]\x1B[0m '
            self._info_msg = '\x1B[32;1m[I]\x1B[0m '
            self._hat = "\x1B[32;1m^\x1B[0m"

    def _init_msg_header(self):
        self._err_msg = "[E] "
        self._warn_msg = "[W] "
        self._db_msg = "[D] "
        self._info_msg = "[I] "

    def info(self, msg, *args, **kwargs):
        _logger_stdout_write(self._info_msg + self.base + (msg % args) + '\n')

    def debug(self, msg, *args, **kwargs):
        _logger_stdout_write(self._db_msg + self.base + (msg % args) + '\n')

    def warning(self, msg, *args, **kwargs):
        _logger_stdout_write(self._warn_msg + self.base + (msg % args) + '\n')
        self.warning_count += 1

    def error(self, msg, *args, **kwargs):
        _logger_stderr_write(self._err_msg + self.base + (msg % args) + '\n')
        self.err_count += 1

    def reset(self):
        self.err_count = 0
        self.warning_count = 0

    def summary(self):
        _logger_stdout_write("%d error(s) generated. %d warning(s) generated.\n" %
                             (self.err_count, self.warning_count))


LOGGER = AIPULogger()


def INFO(msg, *args, **kwargs):
    if _INFO_LEVEL >= LOGLEVEL:
        LOGGER.info(msg, *args, **kwargs)


def DEBUG(msg, *args, **kwargs):
    if _DEBUG_LEVEL >= LOGLEVEL:
        LOGGER.debug(msg, *args, **kwargs)


def WARN(msg, *args, **kwargs):
    if _WARN_LEVEL >= LOGLEVEL:
        LOGGER.warning(msg, *args, **kwargs)


def increase_error():
    global ERROR_COUNT
    ERROR_COUNT += 1


def ERROR(msg, *args, **kwargs):
    increase_error()
    if _ERROR_LEVEL >= LOGLEVEL:
        LOGGER.error(msg, *args, **kwargs)


def FATAL(msg, *args, **kwargs):
    increase_error()
    if _ERROR_LEVEL >= LOGLEVEL:
        LOGGER.error(msg, *args, **kwargs)
        s = traceback.format_exc()
        if s != 'NoneType: None\n':
            LOGGER.error(s)
        sys.exit(-1)


def get_logger():
    return LOGGER


def print_trace_stack():
    traceback.print_exc()


def get_error_count():
    return ERROR_COUNT


def set_log_file(file, err_file=None):
    LOGGER._init_msg_header()
    if err_file is None:
        err_file = file
    _set_logger_file(file, err_file)


def tqdm(*args, **kwargs):
    file = sys.stdout
    if "file" in kwargs:
        file = kwargs["file"]

    disbale = not file.isatty()
    kwargs["disable"] = disbale or kwargs.get("disable", False)
    from tqdm import tqdm as f
    return f(*args, **kwargs)
