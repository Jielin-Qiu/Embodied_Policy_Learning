from schema import *
from preprocess_utils import *
from config.logger_setup import Runtime_Logging
from config.common import *
import time
import sys 
sys.path.append('BLIP/virtualhome1')
sys.path.append('BLIP/virtualhome1/simulation')
sys.path.append('BLIP/virtualhome1/simulation/evolving_graph')
from BLIP.virtualhome1.dataset_utils.execute_script_utils import *
from transformers import pipeline


logger = logging.getLogger(__name__)
logSetup = Runtime_Logging()

if __name__ == "__main__":

    start_time = time.time()
    logSetup.set_log_task("Reading Configuration File")
    logger.info("Reading in details from configuration file")
    config_dir = get_config_path()
    config = load_configuration(config_dir)
    logging_status = config.get('Logging Config', 'status')
    if logging_status != "enable":
        logger.disabled = True

    logSetup.set_log_task("Get Augmented and Non-Augments Scripts and Indexes")
    non_aug_scripts, non_aug_index, aug_scripts, aug_index, non_aug_data_path, aug_data_path, aug_executables, non_aug_executables, non_aug_init, aug_init = get_script_and_indexes(path_to_data, path_to_nonaug_programs, path_to_aug_programs, data_view[1])
    # print(non_aug_data_path)

    logSetup.set_log_task('Get Augmented and Non-Aguemnted Init Graphs')
    logSetup.set_log_task("Establish Model")
    model = pipeline('fill-mask', model='bert-base-uncased')
    logger.info(model)

    logSetup.set_log_task("Prompt Engineer Captions")
    prompt_engineered_aug_scripts, prompt_engineered_non_aug_scripts = prompt_engineer(model, non_aug_scripts, aug_scripts)
    # print(prompt_engineered_aug_scripts)

    logger.info(model)

    logSetup.set_log_task("Create Lists of Data")
    aug_b64, non_aug_b64, indexed_aug_script, indexed_non_aug_script, aug_unique_id, aug_image_id, non_aug_unique_id, non_aug_image_id, non_aug_execs, aug_execs, non_aug_inits, aug_inits = data_lists(non_aug_data_path, aug_data_path, non_aug_index, aug_index, prompt_engineered_non_aug_scripts, prompt_engineered_aug_scripts, aug_executables, non_aug_executables, non_aug_init, aug_init)
    # print(non_aug_inits)
    # print(non_aug_image_id)
    logSetup.set_log_task("Split Data and Save as TSV")
    split_and_list_to_tsv(aug_b64, non_aug_b64, indexed_aug_script, indexed_non_aug_script, aug_unique_id, aug_image_id, non_aug_unique_id, non_aug_image_id, non_aug_execs, aug_execs, non_aug_inits, aug_inits)
                    

                
    



