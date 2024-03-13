# --- PREPROCESSING --- #
test_scenes = ['TrimmedTestScene1_graph', 'TrimmedTestScene2_graph', 'TrimmedTestScene3_graph', 
            'TrimmedTestScene4_graph', 'TrimmedTestScene5_graph', 'TrimmedTestScene6_graph', 'TrimmedTestScene7_graph']
result_scenes = ['results_intentions_march-13-18', 'results_text_rebuttal_specialparsed_programs_turk_july', 'results_text_rebuttal_specialparsed_programs_turk_robot', 
                'results_text_rebuttal_specialparsed_programs_turk_third', 'results_text_rebuttal_specialparsed_programs_upwork_july', 'results_text_rebuttal_specialparsed_programs_upwork_kellensecond', 
                'results_text_rebuttal_specialparsed_programs_upworknturk_second']

render_args = {
    # 'script'                : script,
    'randomize_execution'   : False, 
    'random_seed'           : -1, 
    'processing_time_limit' : 10,
    'skip_execution'        : False, 
    'find_solution'         : True, 
    'output_folder'         : 'Output/', 
    'file_name_prefix'      : "script",
    'frame_rate'            : 25, 
    'image_synthesis'       : ['normal'], 
    'save_pose_data'        : False,
    'image_width'           : 640, 
    'image_height'          : 480, 
    'recording'             : True,
    'save_scene_states'     : False, 
    'camera_mode'           : ['AUTO'], 
    'time_scale'            : 1.0, 
    'skip_animation'        : False
}


render_args1 = {
    'randomize_execution'     : False, 
    'random_seed'             : 1, 
    'processing_time_limit'   : 60,   
    'skip_execution'          : False, 
    'find_solution'           : True, 
    'output_folder'           : 'Output/', 
    'file_name_prefix'        : "script",
    'frame_rate'              : 25, 
    'image_synthesis'         : ['normal'], 
    'capture_screenshot'      : True, 
    'save_pose_data'          : False,
    'image_width'             : 640, 
    'image_height'            : 480, 
    'gen_vid'                 : False,
    'save_scene_states'       : True,
     'character_resource'     : 'Chars/Male1', 
     'camera_mode'            : 'AUTO'
}

results_index = 0
test_index = 0

path_to_images = 'virtualhome1/simulation/unity_simulator/Output'
path_to_aug_programs = 'virtualhome/src/virtualhome/augment_location'
path_to_nonaug_programs = 'virtualhome/src/virtualhome/programs_nonaug_graph'
path_to_aug_init = 'virtualhome/src/virtualhome/augment_location'
path_to_nonaug_init = 'virtualhome/src/virtualhome/programs_nonaug_graph'
path_to_data = 'virtualhome1/simulation/unity_simulator/Data'
data_view = ['auto', 'first_person', 'front_person']


# --- PATHS AND PREPROCESSING --- # 
# data
train_path = 'dataset/train_data.tsv'
val_path = 'dataset/val_data.tsv'
test_path = 'dataset/test_data.tsv'


# paths
result_dir = 'results'
ground_truth_dir = 'ground_truths'
output_dir = 'output_dir'
ground_result_file_path = 'ground_truths/ground_results1.json'
test_result_file_path = 'results/test_results1.json'

ground_result_file_path2 = 'ground_truths/ground_results2.json'
test_result_file_path2 = 'results/test_results2.json'

ground_result_file_path3 = 'ground_truths/ground_results3.json'
test_result_file_path3 = 'results/test_results3.json'

ground_result_file_path4 = 'ground_truths/ground_results4.json'
test_result_file_path4 = 'results/test_results4.json'

ground_result_file_path5 = 'ground_truths/ground_results5.json'
test_result_file_path5 = 'results/test_results5.json'

ground_result_file_path6 = 'ground_truths/ground_results6.json'
test_result_file_path6 = 'results/test_results6.json'

ground_result_file_path7 = 'ground_truths/ground_results7.json'
test_result_file_path7 = 'results/test_results7.json'


# --- BLIP --- #

# set pretrained as a file path or an url
pretrained= 'model_large.pth'
pretrained_embed_model_name = 'bert-base-uncased'

vit= 'large'
vit_grad_ckpt= True
vit_ckpt_layer= 5
batch_size= 8
init_lr= 2e-6

# generation configs
max_length= 15  
min_length= 5
num_beams= 3
prompt= ''

# optimizer
weight_decay= 0.05
min_lr= 0
max_epoch= 20


mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
resolution = 384 


# --- GRIT --- #

# model 
use_gri_feat= True
use_reg_feat= True
grid_feat_dim= 1024
frozen_stages= 2
beam_size= 5
beam_len= 20
dropout= 0.2
attn_dropout= 0.2

vocab_size= 10201
max_len: 54
pad_idx: 1
bos_idx: 2
eos_idx: 3
d_model: 512
n_heads: 8

#grit net
n_memories= 1
n_layers= 3

# detector
checkpoint= '' 
d_model= 512
dim_feedforward= 1024
num_heads= 8
num_layers= 6
num_levels= 4
num_points=4
num_queries= 150
num_classes= 1849
dropout= 0.1
activation= 'relu'
return_intermediate= True
with_box_refine= True