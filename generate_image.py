from distutils import text_file
from preprocess_utils import *
import logging
from config.logger_setup import Runtime_Logging
from config.common import *
import time
from PIL import Image
from schema import *
import sys
sys.path.append('BLIP/virtualhome1')
sys.path.append('BLIP/virtualhome1/simulation')
# from virtualhome.src.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
# import virtualhome.src.virtualhome.simulation.evolving_graph.check_programs as check_programs
# from virtualhome.src.virtualhome.dataset_utils.execute_script_utils import parse_exec_script_file, obtain_scene_id_from_path, obtain_objects_from_message, render_script, render_script_from_path 
# import virtualhome.src.virtualhome.dataset_utils.add_preconds as add_preconds
# from virtualhome.src.virtualhome.dataset_utils.execute_script_utils import *
# from virtualhome.src.virtualhome.simulation.evolving_graph.utils import graph_dict_helper
# from virtualhome.src.virtualhome.simulation.evolving_graph.scripts import read_script_from_list_string, read_script_from_string
from BLIP.virtualhome1.simulation.evolving_graph.scripts import read_script_from_list_string, read_script_from_string
from BLIP.virtualhome1.simulation.evolving_graph.utils import graph_dict_helper
from BLIP.virtualhome1.dataset_utils.execute_script_utils import *
import BLIP.virtualhome1.dataset_utils.add_preconds as add_preconds
import BLIP.virtualhome1.simulation.evolving_graph.check_programs as check_programs
from BLIP.virtualhome1.simulation.unity_simulator.comm_unity import UnityCommunication

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
    path_to_programs = path_to_nonaug_programs

    logSetup.set_log_task("Grabbing Data Dictionary")
    logger.info('Grabbing programs dictionary')
    executable_programs_dic, init_and_final_graphs_dic, initstate_dic, state_list_dic, without_conds_dic = get_iterables_graph(path_to_programs)

    rendered_paths = []

    logSetup.set_log_task("Initializing Test And Result Files")
    test_scene = test_scenes[0]
    for j in range(len(result_scenes)):
        result_scene = result_scenes[j]
        logger.info(test_scene)
        logger.info(result_scene)

        logSetup.set_log_task("Alignment")
        logger.info('Aligning text and json files such that only files with the same names are kept.')
        json_list, txt_list = align_data(path_to_programs, test_scene, result_scene)
        
        for index in tqdm(range(len(txt_list))):
            logSetup.set_log_task("Establishing Data Paths")
            path_to_exec = os.path.join(path_to_programs, f'executable_programs/{test_scene}/{result_scene}/{txt_list[index]}')
            path_to_init_and_final_graphs = os.path.join(path_to_programs, f'init_and_final_graphs/{test_scene}/{result_scene}/{json_list[index]}')
            path_to_initstate = os.path.join(path_to_programs, f'initstate/{result_scene}/{json_list[index]}')
            path_to_state_list = os.path.join(path_to_programs, f'state_list/{test_scene}/{result_scene}/{json_list[index]}')
            path_to_without_conds = os.path.join(path_to_programs, f'withoutconds/{result_scene}/{txt_list[index]}')
            path_to_env = os.path.join(path_to_programs, f'{test_scene}')

            with open(path_to_initstate, 'r') as f:
                preconds = json.load(f)

            action, description, script = parse_withoutconds(path_to_without_conds)
            # action, description, script = parse_exec_script_file(path_to_exec)

            if ('Pick up phone' in action) or ('Watch youtube' in action) or ('Gaze out window' in action):
                logger.info(f'Passed: {txt_list[index]}')
                logger.info(f'Passed: {json_list[index]}')
                pass
            else:
                logger.info(txt_list[index])
                logger.info(json_list[index])
                logger.info(action)
                logger.info(description)
                logger.info(script)

                split_string = action.split()
                file_name = '_'.join(split_string)

                script_name = txt_list[index][:-4]

                render_args  = {
                                'randomize_execution'     : False, 
                                'random_seed'             : 1, 
                                'processing_time_limit'   : 60,   
                                'skip_execution'          : False, 
                                'find_solution'           : True, 
                                'output_folder'           : 'Output/', 
                                'file_name_prefix'        : f"{test_scene}@{result_scene}@{script_name}",
                                'frame_rate'              : 5, 
                                'image_synthesis'         : ['normal'], 
                                'capture_screenshot'      : True, 
                                'save_pose_data'          : False,
                                'image_width'             : 256, 
                                'image_height'            : 256, 
                                'gen_vid'                 : False,
                                'save_scene_states'       : True,
                                'character_resource'     : 'Chars/Male1', 
                                'camera_mode'            : 'FIRST_PERSON'
                            }
                            #PERSON_FRONT, AUTO, FIRST_PERSON
                logger.info(render_args)

                try:
                    # comm.reset()
                    result = render_script_from_path(comm, path_to_exec, path_to_init_and_final_graphs, render_args)
                    logger.info(result)
                    rendered_paths.append(render_args['file_name_prefix'])
                except:
                    pass

    
    with open("virtualhome1/simulation/unity_simulator/new_Data/first_person/render_paths.txt", 'w') as file:
        for row in rendered_paths:
            file.write(row+'\n') 

    total_time = round(time.time() - start_time, 1)
    logger.info(f'Total Runtime: {total_time} seconds')