from configparser import ConfigParser, RawConfigParser
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Text
import logging
logger = logging.getLogger(__name__)


def get_config_path(config_type: Optional[str]="main configuration") -> str:

    this_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(this_dir, os.pardir))

    # Default configuration paths
    configs = {
        'main configuration': {'path': os.path.join(root_dir, 'config'), 'file': 'config.ini'},
        'logging configuration': {'path': os.path.join(root_dir, 'config'), 'file': 'log_config.ini'}
    }

    if config_type not in list(configs.keys()):
        logger.error(f"Configuration for {config_type} is not a valid. Valid configuration files are {', '.join(list(configs.keys()))}")
        sys.exit()

    config_path = os.path.join(
        configs[config_type]['path'], configs[config_type]['file'])

    # Check if configuration file exists.
    if os.path.exists(config_path):
        return config_path
    else:
        logger.error(f"Configuration file at {config_path} does not exist.")
        sys.exit()


def load_configuration(config_path: str, raw: Optional [bool]=False) -> ConfigParser:

    if raw:
        config = RawConfigParser()
    else:
        config = ConfigParser()
    config.read(config_path)

    return config


def write_config_param(payload: dict, config_file: Optional[str]="logging configuration"):

    config_file_path = get_config_path(config_type=config_file)
    parser = load_configuration(config_file_path, raw=True)

    # Loop through each of the section and set the new values
    for group, param in payload.items():
        if group not in parser.sections():
            parser.add_section(group)
        for key, value in param.items():
            parser.set(group, key, value)

    # Save changes to config file
    with open(config_file_path, 'w+') as configfile:
        parser.write(EqualsSpaceRemover(configfile))


class EqualsSpaceRemover:

    output_file = None

    def __init__(self, new_output_file):
        self.output_file = new_output_file

    def write(self, what):
        self.output_file.write(what.replace(" = ", "=", 1))