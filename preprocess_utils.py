from PIL import Image
from io import BytesIO
import base64
from virtualhome.src.virtualhome.simulation.evolving_graph.scripts import read_script, script_to_list_string
import json
import os
import pandas as pd
import numpy as np
from virtualhome.src.virtualhome.simulation.unity_simulator import utils_viz
from config import *
import glob
import sys
sys.path.append('BLIP/virtualhome1')
sys.path.append('BLIP/virtualhome1/simulation')
from BLIP.virtualhome1.dataset_utils.execute_script_utils import *
from schema import *
import uuid 
import csv
from sklearn.model_selection import train_test_split


def parse_withoutconds(path_to_file):


    script_file = read_script(path_to_file)
    list_of_actions = script_to_list_string(script_file)
    action, description = get_action_description(path_to_file)


    return action, description, list_of_actions


def get_iterables_graph(path_to_programs):

    executable_programs_dic = {}
    init_and_final_graphs_dic = {}
    initstate_dic = {}
    state_list_dic = {}
    without_conds_dic = {}
    path_to_executable = os.path.join(path_to_programs, 'executable_programs')
    path_to_init_and_final_graphs = os.path.join(path_to_programs, 'init_and_final_graphs')
    path_to_initstate = os.path.join(path_to_programs, 'initstate')
    path_to_state_list = os.path.join(path_to_programs, 'state_list')
    path_to_withoutcond = os.path.join(path_to_programs, 'withoutconds')
    tested_scenes_paths = [path_to_executable, path_to_init_and_final_graphs, path_to_state_list]
    dics_for_tested = [executable_programs_dic, init_and_final_graphs_dic, state_list_dic]
    results_only_paths = [path_to_initstate, path_to_withoutcond]
    dics_for_results = [initstate_dic, without_conds_dic]
    test_index = 0
    results_index = 0
    for paths in tested_scenes_paths:
        for tests in sorted(os.listdir(paths)):
            if '.DS_Store' not in tests:
                tests_path = os.path.join(paths, tests)
                test_dic = {}
                for results in sorted(os.listdir(tests_path)):
                    if '.DS_Store' not in results:
                        results_path = os.path.join(tests_path, results)
                        file_list = []
                        for file in sorted(os.listdir(results_path)):
                            if '.DS_Store' not in results:
                                file_list.append(file)
                        test_dic[f'{results}'] = file_list
                        # dic_count, file_count = count_files(results_path, file_list)
                        # print(dic_count, file_count)
                dics_for_tested[test_index][f'{tests}'] = test_dic
        test_index +=1
    # print()
    for paths in results_only_paths:
        for results in sorted(os.listdir(paths)):
            if '.DS_Store' not in results:
                results_path = os.path.join(paths, results)
                file_list = []
                for file in sorted(os.listdir(results_path)):
                    if '.DS_Store' not in results:
                        file_list.append(file)
                dics_for_results[results_index][f'{results}'] = file_list
                # dic_count, file_count = count_files(results_path, file_list)
                # print(dic_count, file_count)
        results_index +=1

    return executable_programs_dic, init_and_final_graphs_dic, initstate_dic, state_list_dic, without_conds_dic

def count_files(path_to_dir, file_list):
    count_dir = 0
    count_file = 0
    for _ in file_list:
        count_file += 1
    for _ in sorted(os.listdir(path_to_dir)):
        count_dir +=1
    return count_dir, count_file


def get_action_description(file_name):
    action_description = []
    with open(file_name) as f:
        for line in f:
            if '[' in line:
                break
            line = line.strip()
            action_description.append(line)

    action = action_description[0]
    description = action_description[1:]
    clean_description = []
    for i in description:
        if i != '':
            clean_description.append(i)
        else:
            break
    clean_description = ' '.join(clean_description)

    return action, clean_description


def img_to_base64(path_to_file):
    
    img = Image.open(path_to_file) 
    img_buffer = BytesIO()
    
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str

    return base64_str

def generate_vid(path_to_char_id):

    path_to_vid = 'virtualhome/src/virtualhome/simulation/unity_simulator/Output'
    utils_viz.generate_video(input_path=path_to_vid, prefix='script', char_id=f'{path_to_char_id}', output_path='.')

def align_data(path_to_programs, test_scene = None, results = None):
    executable_programs_dic, init_and_final_graphs_dic, initstate_dic, state_list_dic, without_conds_dic = get_iterables_graph(path_to_programs)
    # print(len(executable_programs_dic[test_scene][results]))
    # print(len(init_and_final_graphs_dic[test_scene][results]))
    # print(len(state_list_dic[test_scene][results]))
    # print(len(initstate_dic[results]))
    # print(len(without_conds_dic[results]))
    json_list = []
    txt_list = []
    intersect_list = []
    intersect_name = []
    exec_name = []
    initstate_name = []
    aligned_names = []
    without_conds_name = []
    initstate = initstate_dic[results]
    without_conds = without_conds_dic[results]
    exec = executable_programs_dic[test_scene][results]

    intersect = set(init_and_final_graphs_dic[test_scene][results]).intersection(state_list_dic[test_scene][results])
    for i in intersect:
        intersect_list.append(i)

    for i in range(len(intersect_list)):
        intersect_name.append(intersect_list[i][:-5])
        exec_name.append(exec[i][:-4])
    intersect2 = set(set(intersect_name).intersection(set(exec_name)))

    for i in range(len(initstate)):
        initstate_name.append(initstate[i][:-5])
        without_conds_name.append(without_conds[i][:-4])

    intersect3 = set(set(initstate_name).intersection(set(without_conds_name)))
    intersect4 = set(intersect2.intersection(intersect3))

    for i in intersect4:
        aligned_names.append(i)

    for i in aligned_names:
        add_json = i + '.json'
        add_txt = i + '.txt'
        json_list.append(add_json)
        txt_list.append(add_txt)

    initstate_dic[results] = json_list
    without_conds_dic[results] = txt_list
    executable_programs_dic[test_scene][results] = txt_list
    state_list_dic[test_scene][results] = json_list
    init_and_final_graphs_dic[test_scene][results] = json_list

    return sorted(json_list), sorted(txt_list)


def remove_empty_dirs(output_path):
    for file in sorted(os.listdir(output_path)):
        if ('.DS_Store' not in file) and ('render_paths.txt' not in file):
            file_path = os.path.join(output_path, file)
            if len(os.listdir(file_path)) == 0:
                os.rmdir(file_path)


def parse_ftaa(path_to_fftaa, parsed_script):
    stripped = []
    action_indexes = []
    distributed_indexes = []
    with open(path_to_fftaa, "r") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            stripped.append(stripped_line)
    for action in stripped:
        indexes = []
        action = action.split(' ')
        if (action[-2] == '1') or (action[-2] == '2'):
            action[-2] = 0
            # indexes.append(action[-2])
        # else:
        indexes.append(int(action[-2]))
        indexes.append(int(action[-1]))
        action_indexes.append(indexes)

    length_of_parsed_script = len(parsed_script)
    start = action_indexes[0][0]
    end = action_indexes[-1][-1]
    add = int(end/length_of_parsed_script)
    for index in range(len(parsed_script)):
        indexes = []

        if index == 0:
            indexes.append(0)
            indexes.append(0+add)
            value = 0+add
        else:
            indexes.append(value+1)
            value_1 = value+1
            indexes.append(value_1 + add)
            value = value_1 + add
        
        
        distributed_indexes.append(indexes)
            

    return stripped, action_indexes, distributed_indexes 

def parse_script(script):
    parsed_script = []
    for action in script:
        action = action.replace('[', '')
        action = action.replace(']', '')
        action = action.replace('<', '')
        action = action.replace('>', '')
        action = action.replace('(', '')
        action = action.replace(')', '')
        action = action.replace('_', ' ')
        action = action.split(' ')
        action = action[:-1]
        action = ' '.join(action)
        parsed_script.append(action)
    return parsed_script


def get_script_and_indexes(path_to_data, path_to_nonaug_programs, path_to_aug_programs, view):
    path_to_view = glob.glob(path_to_data + f'/{view}/*')
    path_to_nonaug_exec = os.path.join(path_to_nonaug_programs, 'executable_programs')
    path_to_aug_exec = os.path.join(path_to_aug_programs, 'executable_programs')
    aug_scripts = []
    non_aug_init = []
    aug_init = []
    non_aug_scripts = []
    aug_index = []
    non_aug_index = []
    non_aug_data_path = []
    aug_data_path = []
    aug_executables = []
    non_aug_executables = []
    for path in sorted(path_to_view):
        split_path = path.split('/')
        if split_path[-1] == 'non_aug':
            for file in sorted(os.listdir(path)):
                if ('render_paths.txt' not in file) and ('.DS_Store' not in file):
                    split_file = file.split('@')
                    split_file[-1] = split_file[-1] + '.txt'
                    
                    joined_file = '/'.join(split_file)
                    program_path = os.path.join(path_to_nonaug_exec, joined_file)
                    data_path = os.path.join(path, file)
                    split_program = program_path.split('/')
                    split_program[4] = 'init_and_final_graphs'
                    split_program[-1] = split_program[-1][:-4] + '.json'
                    rendered_init = '/'.join(split_program)

                    action, description, script = parse_exec_script_file(program_path)
                    if not os.path.exists(data_path):
                        raise Exception('Path Does Not Exist')
                    # print('script: ', script)
                    # print('ftaa script: ', ftaa_script)
                    parsed_script = parse_script(script)
                    ftaa_script, action_indexes, distributed_indexes = parse_ftaa(os.path.join(data_path, f'ftaa_{file}.txt'), parsed_script)
                    action_index_limit = action_indexes[-1][-1]
                    distributed_indexes[-1][-1] = action_index_limit
                    # print(parsed_script)
                    # print(distributed_indexes)
                    # if len(parsed_script) != len(distributed_indexes):
                    #     print('not equal')
                    non_aug_init.append(rendered_init)
                    non_aug_scripts.append(parsed_script)
                    non_aug_index.append(distributed_indexes)
                    non_aug_data_path.append(data_path)
                    non_aug_executables.append(script)

        else:
            for file in sorted(os.listdir(path)):
                if ('render_paths.txt' not in file) and ('.DS_Store' not in file):
                    split_file = file.split('@')
                    joined_file = '/'.join(split_file)
                    program_path = os.path.join(path_to_aug_exec, joined_file)
                    data_path = os.path.join(path, file)
                    split_program = program_path.split('/')
                    split_program[4] = 'init_and_final_graphs'
                    split_program[-1] = split_program[-1][:-4] + '.json'
                    rendered_init = '/'.join(split_program)
                    if not os.path.exists(data_path):
                        raise Exception('Path Does Not Exist')
   
                    action, description, script = parse_exec_script_file(program_path)
                    parsed_script = parse_script(script)
                    ftaa_script, action_indexes, distributed_indexes = parse_ftaa(os.path.join(data_path, f'ftaa_{file}.txt'), parsed_script)
                    action_index_limit = action_indexes[-1][-1]
                    distributed_indexes[-1][-1] = action_index_limit
                    aug_init.append(rendered_init)
                    aug_scripts.append(parsed_script)
                    aug_index.append(distributed_indexes)
                    aug_data_path.append(data_path)
                    aug_executables.append(script)
    return non_aug_scripts, non_aug_index, aug_scripts, aug_index, non_aug_data_path, aug_data_path, aug_executables, non_aug_executables, non_aug_init, aug_init


def preprocess_prompts(prompt):
    lower_cased = prompt.lower()
    split_lower = lower_cased.split()
    split_lower.insert(1, '[MASK]')
    # print(split_lower)
    if split_lower[0] == 'turnto':
        split_lower[0] = 'turn to'
    elif split_lower[0] == 'switchon':
        split_lower[0] = 'switch on'
    elif split_lower[0] == 'lookat':
        split_lower[0] = 'look at'
    elif split_lower[0] == 'putobjback':
        split_lower[0] = 'put object back'
    elif split_lower[0] == 'switchoff':
        split_lower[0] = 'switch off'
    elif split_lower[0] == 'pointat':
        split_lower[0] = 'point at'
    joined = ' '.join(split_lower)
    return joined


    
def prompt_engineer(model, non_aug_scripts, aug_scripts):
    prompt_engineered_non_aug_scripts = []
    prompt_engineered_aug_scripts = []
    for script in non_aug_scripts:
        # print(script)
        script_list = []
        for prompt in script:
            preprocessed_prompt = preprocess_prompts(prompt)
            pred = model(preprocessed_prompt)
            best_score = pred[0]
            pred_sequence = best_score['sequence']
            script_list.append(pred_sequence)
        prompt_engineered_non_aug_scripts.append(script_list)

    for script in aug_scripts:
        # print(script)
        script_list = []
        for prompt in script:
            preprocessed_prompt = preprocess_prompts(prompt)
            pred = model(preprocessed_prompt)
            best_score = pred[0]
            pred_sequence = best_score['sequence']
            script_list.append(pred_sequence)
        prompt_engineered_aug_scripts.append(script_list)
    return prompt_engineered_aug_scripts, prompt_engineered_non_aug_scripts



def data_lists(non_aug_data_path, aug_data_path, non_aug_index, aug_index, non_aug_scripts, aug_scripts, aug_executables, non_aug_executables,non_aug_init, aug_init):
    aug_b64 = []
    non_aug_b64 = []
    indexed_aug_script = []
    indexed_non_aug_script = []
    aug_unique_id = []
    aug_image_id = []
    non_aug_unique_id = []
    non_aug_image_id = []
    aug_execs = []
    non_aug_execs = []
    non_aug_inits = []
    aug_inits=[]
    count_path = 0
    for path in tqdm(non_aug_data_path, desc = 'Iterating paths to images: '):
        count = 0
        count_index = 0
        for image in sorted(os.listdir(path)):
            if ('.txt' not in image) and ('.DS' not in image) and ('.json' not in image):
                # print(path)
                # print(image)
                path_to_image = os.path.join(path, image)
                base64_str = img_to_base64(path_to_image)
                non_aug_b64.append(base64_str)
                iter_indexes = non_aug_index[count_path]
                # print(iter_indexes)
                if count == iter_indexes[0][1]:
                    count_index +=1
                indexed_non_aug_script.append(non_aug_scripts[count_path][count_index])
                non_aug_execs.append(non_aug_executables[count_path][count_index])
                non_aug_unique_id.append(uuid.uuid1())
                non_aug_image_id.append(path_to_image)
                non_aug_inits.append(non_aug_init[count_path])
                count +=1
        count_path +=1

    count_path = 0
    for path in tqdm(aug_data_path, desc = 'Iterating paths to images: '):
        count = 0
        count_index = 0
        for image in sorted(os.listdir(path)):
            if ('.txt' not in image) and ('.DS' not in image) and ('.json' not in image):
                path_to_image = os.path.join(path, image)
                base64_str = img_to_base64(path_to_image)
                aug_b64.append(base64_str)
                iter_indexes = aug_index[count_path]
                # print(iter_indexes)
                if count == iter_indexes[0][1]:
                    count_index +=1
                indexed_aug_script.append(aug_scripts[count_path][count_index])
                aug_execs.append(aug_executables[count_path][count_index])
                aug_unique_id.append(uuid.uuid1())
                aug_image_id.append(path_to_image)
                aug_inits.append(aug_init[count_path])
                count +=1
        count_path +=1
    return aug_b64, non_aug_b64, indexed_aug_script, indexed_non_aug_script, aug_unique_id, aug_image_id, non_aug_unique_id, non_aug_image_id, non_aug_execs, aug_execs, non_aug_inits, aug_inits


def split_and_list_to_tsv(aug_b64, non_aug_b64, indexed_aug_script, indexed_non_aug_script, aug_unique_id, aug_image_id, non_aug_unique_id, non_aug_image_id, non_aug_execs, aug_execs, non_aug_inits, aug_inits):
    b64 = aug_b64 + non_aug_b64
    indexed_scripts = indexed_aug_script + indexed_non_aug_script
    unique_id = aug_unique_id + non_aug_unique_id
    image_id = aug_image_id + non_aug_image_id
    execs = aug_execs + non_aug_execs
    inits = aug_inits + non_aug_inits
    len_of_data = len(b64)
    pred_object = [''] * len_of_data
    train_b64, test_b64, train_indexed_scripts, test_indexed_scripts, train_unique_id, test_unique_id, train_image_id, test_image_id, train_pred_object, test_pred_object, train_execs, test_execs, train_inits, test_inits = train_test_split(b64, indexed_scripts, unique_id, image_id, pred_object, execs, inits, random_state = 2, shuffle = True, test_size = 0.3)
    test_b64, val_b64, test_indexed_scripts, val_indexed_scripts, test_unique_id, val_unique_id, test_image_id, val_image_id, test_pred_object, val_pred_object, test_execs, val_execs, test_inits, val_inits = train_test_split(test_b64, test_indexed_scripts, test_unique_id, test_image_id, test_pred_object, test_execs, test_inits, shuffle = True, random_state = 2, test_size = 0.5)
    train_data = zip(train_unique_id, train_image_id, train_indexed_scripts, train_pred_object, train_b64, train_execs, train_inits)
    with open('train_data.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for ui, iid, idx, pred, b, e, init in train_data:
            tsv_output.writerow([ui, iid, idx, pred, b, e, init ])
    val_data = zip(val_unique_id, val_image_id, val_indexed_scripts, val_pred_object, val_b64, val_execs, val_inits)
    with open('val_data.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for ui, iid, idx, pred, b, e, init in val_data:
            tsv_output.writerow([ui, iid, idx, pred, b, e, init ])
    test_data = zip(test_unique_id, test_image_id, test_indexed_scripts, test_pred_object, test_b64, test_execs, test_inits)
    with open('test_data.tsv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for ui, iid, idx, pred, b, e, init in test_data:
            tsv_output.writerow([ui, iid, idx, pred, b, e, init ])


def open_file(path_to_file):
    if os.path.splitext(path_to_file)[1] == '.json':
        json_file = open(path_to_file)
        json_dic = json.load(json_file)
        return json_dic
    else:
        txt_file = open(path_to_file, 'r')
        txt_file = txt_file.read()
        txt_file_list = txt_file.split('\n')
        return txt_file_list

def get_init_graphs(path_to_rendered_paths):
    base_path = 'virtualhome/src/virtualhome/'

    init_graphs = {
        'FRONT_PERSON' : [],
        'FIRST_PERSON' : [],
        'AUTO'         : []
    }

    for rendered_path in path_to_rendered_paths:
        rendered_file = open_file(rendered_path)
        if '' in rendered_file:
            rendered_file.remove('')
        
        for file in rendered_file:
            split_file = file.split('@')
            if '.txt' in file:
                split_file = split_file[:-1]
            split_file[-1] = split_file[-1] + '.json'

            joined_file = '/'.join(split_file)

            if 'non_aug' in rendered_path:
                path_to_graphs = f'{base_path}programs_nonaug_graph/'
                if 'front_person' in rendered_path:
                    init_graphs['FRONT_PERSON'].append(path_to_graphs + joined_file)
                elif 'first_person' in rendered_path:
                    init_graphs['FIRST_PERSON'].append(path_to_graphs + joined_file)
                else:
                    init_graphs['AUTO'].append(path_to_graphs + joined_file)
            else:
                path_to_graphs = f'{base_path}augment_location/'
                if 'front_person' in rendered_path:
                    init_graphs['FRONT_PERSON'].append(path_to_graphs+ joined_file)
                elif 'first_person' in rendered_path:
                    init_graphs['FIRST_PERSON'].append(path_to_graphs + joined_file)
                else:
                    init_graphs['AUTO'].append(path_to_graphs + joined_file)
    return init_graphs


def parse_out(out_list):

    preprocessed_scripts = []
    graphs_paths = []
    for i in out_list:
        script = i['caption']
        graphs_paths.append(f"../{i['graph']}")
        split_script = script.split(' ')
        split_script = [f'[{split_script[1].upper()}] <{split_script[4]}> (1)']
        preprocessed_scripts.append(split_script)

    return preprocessed_scripts, graphs_paths