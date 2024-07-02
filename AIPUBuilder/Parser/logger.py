# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


# cython: language_level=3
# !/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import logging
import traceback
try:
    from AIPUBuilder import __release__
except ImportError:
    __release__ = True

__all__ = [
    'LOGGER',
    'INFO',
    'WARN',
    'WARN_EXCEPTION',
    'DEBUG',
    'ERROR',
    'FATAL',
    'get_error_count',
    'init_logging'
]


class CUPLogger():
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.warning_count = 0
        self.err_count = 0
        self.header = '[Parser]: '
        self._err_msg = '[E] '
        self._warn_msg = '[W] '
        self._db_msg = '[D] '
        self._info_msg = '[I] '

    def set_color(self, colorful=True):
        if colorful and os.name != 'nt' and sys.stdout.isatty() and sys.stderr.isatty():
            self._err_msg = '\x1B[31;1m[E]\x1B[0m '
            self._warn_msg = '\x1B[34;1m[W]\x1B[0m '
            self._db_msg = '\x1B[33;1m[D]\x1B[0m '
            self._info_msg = '\x1B[32;1m[I]\x1B[0m '
        else:
            self._err_msg = '[E] '
            self._warn_msg = '[W] '
            self._db_msg = '[D] '
            self._info_msg = '[I] '

    def apply_header(self, msg):
        if not msg.startswith(self.header):
            msg = self.header + msg
        return msg

    def show_as_is(self, msg, *args, **kwargs):
        self.logger.info(msg % args)

    def info(self, msg, *args, **kwargs):
        self.logger.info(self._info_msg + (self.apply_header(msg) % args))

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(self._db_msg + (self.apply_header(msg) % args))

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(self._warn_msg + (self.apply_header(msg) % args))
        self.warning_count += 1

    def error(self, msg, *args, **kwargs):
        self.logger.error(self._err_msg + (self.apply_header(msg) % args))
        self.err_count += 1

    def fatal(self, msg, *args, **kwargs):
        self.logger.critical(self._err_msg + (self.apply_header(msg) % args))
        self.err_count += 1

    def summary(self):
        self.logger.info(self._info_msg + self.header + ('%d error(s), %d warning(s) generated.' %
                                                         (self.err_count, self.warning_count)))


LOGGER = CUPLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')


def INFO(msg, *args, **kwargs):
    LOGGER.info(msg, *args, **kwargs)


def DEBUG(msg, *args, **kwargs):
    LOGGER.debug(msg, *args, **kwargs)


def WARN(msg, *args, **kwargs):
    LOGGER.warning(msg, *args, **kwargs)


def WARN_EXCEPTION(msg, *args, **kwargs):
    if __release__:
        WARN(msg, *args, **kwargs)
    else:
        ERROR(msg, *args, **kwargs)


def ERROR(msg, *args, **kwargs):
    LOGGER.error(msg, *args, **kwargs)
    exc = traceback.format_exc().replace('%', '%%')
    if exc != 'NoneType: None\n':
        LOGGER.show_as_is(exc)
    else:
        stacks = traceback.format_stack(limit=2)
        LOGGER.show_as_is('Error comes from: \n' + stacks[0].replace('%', '%%'))


def FATAL(msg, *args, **kwargs):
    LOGGER.fatal(msg, *args, **kwargs)
    exc = traceback.format_exc().replace('%', '%%')
    if exc != 'NoneType: None\n':
        LOGGER.show_as_is(exc)
    LOGGER.error('Parser Failed!')
    sys.exit(-1)


def get_error_count():
    LOGGER.summary()
    return LOGGER.err_count


def init_logging(verbose, logfile=None):
    if verbose:
        logging_level = logging.DEBUG
        lib_logger = logging.getLogger('onnxscript')
        lib_logger.setLevel(logging.ERROR)
    else:
        logging_level = logging.INFO
    log_format = '%(message)s'
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging_level, format=log_format, force=True)
        LOGGER.set_color(False)
    else:
        logging.basicConfig(level=logging_level, format=log_format, force=True)
        LOGGER.set_color(True)
