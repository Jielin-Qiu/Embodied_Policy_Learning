from distutils import text_file
from utils import *
import logging
from config.logger_setup import Runtime_Logging
from config.common import *
import time
from PIL import Image
from schema import *
# from virtualhome.src.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
# import virtualhome.src.virtualhome.simulation.evolving_graph.check_programs as check_programs
# from virtualhome.src.virtualhome.dataset_utils.execute_script_utils import parse_exec_script_file, obtain_scene_id_from_path, obtain_objects_from_message, render_script, render_script_from_path 
# import virtualhome.src.virtualhome.dataset_utils.add_preconds as add_preconds
# from virtualhome.src.virtualhome.dataset_utils.execute_script_utils import *
# from virtualhome.src.virtualhome.simulation.evolving_graph.utils import graph_dict_helper
# from virtualhome.src.virtualhome.simulation.evolving_graph.scripts import read_script_from_list_string, read_script_from_string
from virtualhome1.simulation.evolving_graph.scripts import read_script_from_list_string, read_script_from_string
from virtualhome1.simulation.evolving_graph.utils import graph_dict_helper
from virtualhome1.dataset_utils.execute_script_utils import *
import virtualhome1.dataset_utils.add_preconds as add_preconds
import virtualhome1.simulation.evolving_graph.check_programs as check_programs
from virtualhome1.simulation.unity_simulator.comm_unity import UnityCommunication

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

    logSetup.set_log_task('Connecting With Unity Engine')
    comm = UnityCommunication()
    logger.info(comm)

    logSetup.set_log_task('Get Graph Dictionary Helper')
    helper = graph_dict_helper()
    logger.info(helper)


    logSetup.set_log_task("Establishing Root Data Path")
    path_to_programs = path_to_aug_programs

    logSetup.set_log_task('Establishing paths to Executables and init graphs')
    path_to_exec = sorted(glob.glob(path_to_aug_programs + f'/executable_programs/*/*/*/*'))
    path_to_init_and_final_graphs  = sorted(glob.glob(path_to_aug_programs + f'/init_and_final_graphs/*/*/*/*'))

    rendered_paths = []

    
    for j in tqdm(range(len(path_to_exec))):

        split_exec = path_to_exec[j].split('/')
        trimmed_name = split_exec[5]
        result_name = split_exec[6]
        file_name = split_exec[-2]
        index_name = split_exec[-1]

        # action, description, script = parse_withoutconds(path_to_without_conds)
        action, description, script = parse_exec_script_file(path_to_exec[j])

        if ('Pick up phone' in action) or ('Watch youtube' in action) or ('Gaze out window' in action):
            logger.info(f'Passed: {file_name}')
            pass
        else:

            logger.info(trimmed_name)
            logger.info(result_name)
            logger.info(file_name)
            logger.info(action)
            logger.info(description)
            logger.info(script)

            split_string = action.split()
            action_name = '_'.join(split_string)

            render_args  = {
                            'randomize_execution'     : False, 
                            'random_seed'             : 1, 
                            'processing_time_limit'   : 60,   
                            'skip_execution'          : False, 
                            'find_solution'           : True, 
                            'output_folder'           : 'Output/', 
                            'file_name_prefix'        : f"{trimmed_name}@{result_name}@{file_name}@{index_name}",
                            'frame_rate'              : 5, 
                            'image_synthesis'         : ['normal'], 
                            'capture_screenshot'      : True, 
                            'save_pose_data'          : False,
                            'image_width'             : 256, 
                            'image_height'            : 256, 
                            'gen_vid'                 : False,
                            'save_scene_states'       : True,
                            'character_resource'     : 'Chars/Male1', 
                            'camera_mode'            : 'PERSON_FRONT'
                        }
            logger.info(render_args)
            #PERSON_FRONT, AUTO, FIRST_PERSON

            try:
                # comm.reset()
                result = render_script_from_path(comm, path_to_exec[j], path_to_init_and_final_graphs[j], render_args)
                logger.info(result)
                rendered_paths.append(render_args['file_name_prefix'])
            except:
                pass

    with open("virtualhome1/simulation/unity_simulator/new_Data/front_person/render_paths.txt", 'w') as file:
        for row in rendered_paths:
            file.write(row+'\n') 

    total_time = round(time.time() - start_time, 1)
    logger.info(f'Total Runtime: {total_time} seconds')