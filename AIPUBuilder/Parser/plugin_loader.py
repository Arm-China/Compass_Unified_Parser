# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2024 Arm Technology (China) Co. Ltd.


import importlib
import pkgutil
import functools

from enum import Enum, unique
import os
import sys
from .logger import WARN
AIPUPLUGIN_PATH = os.environ.get('AIPUPLUGIN_PATH', '').split(':')

PARSER_OP_DICT = dict()

PLUGIN_PREFIX = 'aipubt_'


class PluginTypeError(Exception):
    pass


@unique
class PluginType(Enum):
    Builder = 0x314
    Checker = 0x315
    Parser = 0x200


PLUGINS = {t: dict() for t in PluginType}
VERSIONS = {t: dict() for t in PluginType}


def register_plugin(type=PluginType.Parser, version=0):
    '''
    register a plugin with type and version
    supporting plugin type: Parser,Builder
    '''

    global PLUGINS, VERSIONS

    def tofloat(str_version):
        try:
            fval = float(str_version)
        except:
            split_version = str_version.strip(' ').split('.')
            fval = float('.'.join([split_version[0], ''.join(split_version[1:])]))
        return fval

    def wrapper(cls):
        if type not in PluginType:
            raise PluginTypeError('Unsupported Plugin Type.')
        elif type == PluginType.Parser:
            from .plugin_op import ParserOp
            if issubclass(cls, ParserOp):
                optype = cls.op_type
                if optype not in PARSER_OP_DICT:
                    if hasattr(cls, '_check_'):
                        if not cls._check_():
                            pass
                        else:
                            if hasattr(cls, '_subgraph_type') and cls._subgraph_type == 'named_subgraph':
                                # for named plugin, the add a prefix for register optype
                                PARSER_OP_DICT['.named_subgraph.' + optype] = cls
                            elif hasattr(cls, 'input_nodes') and cls.input_nodes is not None:
                                preprocess_optype = '.preprocess.' + optype
                                if preprocess_optype in PARSER_OP_DICT:
                                    WARN(
                                        'Plugin with op_type %s is overriden by class %s!' % (optype, cls.__name__))
                                PARSER_OP_DICT[preprocess_optype] = cls
                    PARSER_OP_DICT[optype] = cls
                    PARSER_OP_DICT[optype.upper()] = cls
            else:
                raise TypeError(
                    'Parser Plugin must be a subclass of ParserOp. But the plugin\'s class is %s' % cls.__name__)

        if cls.__name__ not in PLUGINS[type]:
            PLUGINS[type][cls.__name__] = cls
            VERSIONS[type][cls.__name__] = str(version)
        return cls
    return wrapper


AIPUPLUGIN_PATH = [i for i in set(AIPUPLUGIN_PATH + ['./plugin', '.']) if len(i) != 0]
sys.path = AIPUPLUGIN_PATH + sys.path

ALL_MODULES = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in pkgutil.iter_modules(AIPUPLUGIN_PATH)
    if name.startswith(PLUGIN_PREFIX)
}
