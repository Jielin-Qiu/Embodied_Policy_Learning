import pandas as pd
import numpy as np
import os
from virtualhome.src.virtualhome.simulation.evolving_graph.scripts import read_script, script_to_list_string, Script
from virtualhome.src.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.src.virtualhome.simulation.evolving_graph import utils
from virtualhome.src.virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome.src.virtualhome.simulation.evolving_graph.environment import EnvironmentGraph

from utils import *
import json




if __name__ == '__main__':
    

    properties_data = utils.load_properties_data()
    object_states = utils.load_object_states()
    object_placing = utils.load_object_placing()

    action_list, description_list, scripts_list, initstate_list = get_iterables()
    comm = UnityCommunication()


    for i in range(len(action_list)):

        action = action_list[i]
        description = description_list[i]
        script = scripts_list[i]
        initstate = initstate_list[i]
        new_script = []
        for i in script:
            i = f'<char1> {i}'
            new_script.append(i)
        # print(new_script)
        # for i in range(len(script)):
        #     print(script[i])

        print('Starting scene...')
        comm.reset()
        graph_file = 'file7_1.json'
        with open('virtualhome/src/virtualhome/programs_nonaug_graph/init_and_final_graphs/TrimmedTestScene1_graph/results_intentions_march-13-18/{}'.format(graph_file), 'r') as f:
            graphs = json.load(f)
            first_graph = graphs['init_graph']
        helper = utils.graph_dict_helper(properties_data, object_placing, object_states, max_nodes=15)
        helper.initialize(first_graph)
        s, cam_count = comm.camera_count()
        path_to_withoutconds= 'virtualhome/src/virtualhome/programs_nonaug_graph/executable_programs/TrimmedTestScene1_graph/results_intentions_march-13-18/file7_1.txt'
        script = read_script(path_to_withoutconds)
        list_string = script_to_list_string(script)
        new_script = []
        for i in script:
            i = f'<char1> {i}'
            new_script.append(i)

        precond_path = f'virtualhome/src/virtualhome/programs_nonaug_graph/initstate/results_intentions_march-13-18/{graph_file}'
        with open(precond_path, 'r') as f:
            precond = json.load(f)
        comm.expand_scene(first_graph)       
       # print(script[0])
        # print(cam_count)
        # s, images = comm.camera_image([0, cam_count-1])

        comm.add_character('Chars/Male2')
        # script = [
        #     '<char1> [Walk] <chair> (1)',
        #     '<char1> [Sit] <chair> (1)'
        # ]
        comm.render_script(new_script, find_solution=True, recording=True)
        # print('Generating video...')
        # comm.render_script(script, recording = True, image_height=256, image_width=256, find_solution = True)
        # s, graph = comm.environment_graph()
        # print(s)
        # print()
        # print(graph.keys())

        # success, message = comm.render_script(script=script,
                                    #   processing_time_limit=60,
                                    #   find_solution=False,
                                    #   image_synthesis=[])
        
        break

    
    

    # --- make sure we have the same number of instances for scripts, actions, descriptions, and initstates
    # print(len(action_list), len(description_list), len(scripts_list), len(initstate_list))   
    
    # --- NOTES
    # 2807 files for initstate and withoutconds
    # 2020 actions, descriptions, inistate, and scripts after cleaning



        