import os
from . import common
from datetime import datetime, timedelta
import time

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
class Runtime_Logging(object):

    def __init__(self):
        # Get the filepath to configuration file and create config object
        config_file_path = common.get_config_path(config_type="logging configuration")
        config = common.load_configuration(config_file_path)

        self.log_output_file = config.get('Logging Config', 'output_file')
        # Get root folder path of project so we can build folder path to the logs folder.
        root_dir = os.path.abspath(os.path.join(config_file_path, os.pardir, os.pardir)) 
        self.log_file_path = os.path.join(root_dir, 'logs', self.log_output_file)

        self.log_file_folder = os.path.join(root_dir, 'logs')
        self.run_id = int(config.get('Logging Config', 'last_run_id'))
        self.max_log_filesize_megabytes = int(config.get('Logging Config', 'max_log_filesize_mb'))*1000000
        self.backup_count = int(config.get('Logging Config', 'backup_count'))
        self.logging_level = int(config.get('Logging Config', 'level'))
        
        # Properties based on last runtime.
        self.last_start_time = config.get('Logging Config', 'last_start_time')
        self.last_end_time = config.get('Logging Config', 'last_end_time')


        # Properties based on current runtime
        self.event_details = self.extract_last_event_details()

        # Initialize logging properties
        self._setup_logging()


    def get_final_log_runtime(self):
        if type(self.last_start_time) is float:
            self.last_start_time = datetime.fromtimestamp(int(self.last_start_time)).strftime('%Y-%m-%d %I:%M:%S %p')
            self.last_end_time = datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %I:%M:%S %p')


    def _setup_logging(self):
        """ Initialize the logger properties with handlers and configuration settings.
            This method is initiated when runtime logging object is created.
        """
        # Create logging directory if needed
        if not os.path.exists(self.log_file_folder):
            os.makedirs(self.log_file_folder)

        # Configure the logger formatter
        log_message_format = '%(asctime)s.%(msecs)03d | %(levelname)-8s | RUN %(run_id)-10s | Task - %(current_task)-40s | %(funcName)-30s |  %(message)-250s'

        # Determine the run ID for use in the logger
        # If it's the first run, increment run id by 1.
        # This will be the initial time variable used to query content from event feeds.
        self.run_id += 1
        
        handler = RotatingFileHandler(filename=self.log_file_path, maxBytes=self.max_log_filesize_megabytes, backupCount=self.backup_count)
        logging.basicConfig(level=self.logging_level, handlers=[handler] ,format=log_message_format, datefmt='%Y-%m-%d %H:%M:%S')

        # Begin Logging by writing header and classify all tasks as 'Initialize'
        self.set_log_task("Initialize")
        # self.write_log_header()


    def set_log_task(self, current_task: str):

        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.run_id = self.run_id
            record.current_task = current_task
            return record
        logging.setLogRecordFactory(record_factory)


    def write_log_header(self):
    
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.log_file_path, 'a') as file:
            file.write(f"-- Run {self.run_id} -- {timestamp} \n")

    
    def write_to_config(self):
        config_dict = dict()
        config_dict['Logging Config'] = dict()
        config_dict['Logging Config']['max_log_filesize_mb'] = str(int(self.max_log_filesize_megabytes/1000000))
        config_dict['Logging Config']['output_file'] = self.log_output_file
        config_dict['Logging Config']['backup_count'] = str(self.backup_count)
        config_dict['Logging Config']['level'] = str(self.logging_level)
        config_dict['Logging Config']['last_run_id'] = str(self.run_id)
        config_dict['Logging Config']['last_start_time'] = str(self.last_start_time)
        config_dict['Logging Config']['last_end_time'] = str(self.last_end_time)

        # Grab event_details dictionary property and add to config_dict
        if len(self.event_details) > 0:
            config_dict.update(self.event_details)

        common.write_config_param(config_dict)


    def extract_last_event_details(self) -> dict:

        # Get the filepath to configuration file and create config object
        config_file_path = common.get_config_path(config_type="logging configuration")
        config = common.load_configuration(config_file_path)

        event_configLog_dict = dict()
        for section in config.sections():
            if 'Logging Config' not in section:
                event_configLog_dict[section] = dict()
                event_configLog_dict[section]['name'] = config.get(section, 'name')
                event_configLog_dict[section]['item_id'] = config.get(section, 'item_id')
                event_configLog_dict[section]['last_query_timestamp'] = float(config.get(section, 'last_query_timestamp'))
                event_configLog_dict[section]['query_status'] = config.get(section, 'query_status')
        if len(event_configLog_dict) > 0:
            return event_configLog_dict
        else:
            return dict() 


    def extract_last_run_from_log(self) -> list:
  
        back_up_log_count = self.backup_count
        header_pattern = f"-- Run {self.run_id} --"
        log_file_path = self.log_file_path
        
        def extract_from_log(log_file_path: str) -> list:

            header_pattern = f"-- Run {self.run_id} --"
            pattern = f"RUN {self.run_id}"
            lines = list()
            with open (log_file_path) as myfile:
                for line in myfile:
                    if pattern in line or header_pattern in line:
                        lines.append(line.rstrip('\n'))
            return lines
        log_run_lines = extract_from_log(self.log_file_path)
        if header_pattern not in log_run_lines[0]:
            backuplog_lines = list()
            for backup_count in range(self.backup_count):
                if os.path.isfile(f'{self.log_file_path}{backup_count+1}'):
                    logger.debug(f'Reading through {self.log_file_path}{backup_count+1} to find header pattern: {header_pattern}')
                    backuplog_run = extract_from_log(f'{self.log_file_path}{backup_count+1}')
                    if backuplog_run:
                        if header_pattern not in backuplog_run[0]:
                            continue
                    else:
                        backuplog_lines = backuplog_run
                        break
            return backuplog_lines[1:] + log_run_lines
        else:
            return log_run_lines[1:]