from schema import *
from preprocess_utils import *
import sys
sys.path.append('virtualhome1')
sys.path.append('virtualhome1/simulation')
sys.path.append('../BLIP')
import glob
from virtualhome1.simulation.evolving_graph.scripts import read_script_from_list_string, read_script_from_string
from virtualhome1.simulation.evolving_graph.utils import graph_dict_helper
from virtualhome1.dataset_utils.execute_script_utils import *
import virtualhome1.dataset_utils.add_preconds as add_preconds
import virtualhome1.simulation.evolving_graph.check_programs as check_programs
from virtualhome1.simulation.unity_simulator.comm_unity import UnityCommunication
from tqdm import tqdm


if __name__ == '__main__':


    results = open_file('results/all_results.json')
    preprocessed_scripts, graph_paths = parse_out(results)

    comm = UnityCommunication()
    successes = 0
    total = 0
    env_1_success = 0
    env_2_success = 0
    env_3_success = 0
    env_4_success = 0
    env_5_success = 0
    env_6_success = 0
    env_7_success = 0
    env_1_total = 0
    env_2_total = 0
    env_3_total = 0
    env_4_total = 0
    env_5_total = 0
    env_6_total = 0
    env_7_total = 0

    for i in tqdm(range(len(preprocessed_scripts)), desc = 'Rendering Outputs'):
        script  = preprocessed_scripts[i]
        graph = graph_paths[i]
        render_args  = {
                        'randomize_execution'     : False, 
                        'random_seed'             : 1, 
                        'processing_time_limit'   : 60,   
                        'skip_execution'          : False, 
                        'find_solution'           : True, 
                        'output_folder'           : 'Output/', 
                        'file_name_prefix'        : f"{graph}@{script}",
                        'frame_rate'              : 5, 
                        'image_synthesis'         : None, 
                        'capture_screenshot'      : True, 
                        'save_pose_data'          : False,
                        'image_width'             : 256, 
                        'image_height'            : 256, 
                        'gen_vid'                 : False,
                        'save_scene_states'       : False,
                        'character_resource'     : 'Chars/Male1', 
                        'camera_mode'            : 'FIRST_PERSON'
                    }
        try:
            comm.reset()
            result = render_script_from_path(comm, script, graph, render_args)
            if result['success_exec'] == True:
                successes +=1
                total+=1

                if 'TrimmedTestScene1_graph' in graph:
                    env_1_success +=1
                    env_1_total +=1
                elif 'TrimmedTestScene2_graph' in graph:
                    env_2_success +=1
                    env_2_total +=1
                elif 'TrimmedTestScene3_graph' in graph:
                    env_3_success +=1
                    env_3_total +=1
                elif 'TrimmedTestScene4_graph' in graph:
                    env_4_success +=1
                    env_4_total +=1
                elif 'TrimmedTestScene5_graph' in graph:
                    env_5_success +=1
                    env_5_total +=1
                elif 'TrimmedTestScene6_graph' in graph:
                    env_6_success +=1
                    env_6_total +=1
                else: 
                    env_7_success +=1
                    env_7_total +=1
            else:
                total+=1
                if 'TrimmedTestScene1_graph' in graph:
                    env_1_total +=1
                elif 'TrimmedTestScene2_graph' in graph:
                    env_2_total+=1
                elif 'TrimmedTestScene3_graph' in graph:
                    env_3_total+=1
                elif 'TrimmedTestScene4_graph' in graph:
                    env_4_total +=1
                elif 'TrimmedTestScene5_graph' in graph:
                    env_5_total +=1
                elif 'TrimmedTestScene6_graph' in graph:
                    env_6_total +=1
                else: 
                    env_7_total +=1
        except:
            print('Error')

    print(f'Overall: {successes/total}')
    print(f'Environment 1: {env_1_success/env_1_total}')
    print(f'Environment 2: {env_2_success/env_2_total}')
    print(f'Environment 3: {env_3_success/env_3_total}')
    print(f'Environment 4: {env_4_success/env_4_total}')
    print(f'Environment 5: {env_5_success/env_5_total}')
    print(f'Environment 6: {env_6_success/env_6_total}')
    print(f'Environment 7: {env_7_success/env_7_total}')