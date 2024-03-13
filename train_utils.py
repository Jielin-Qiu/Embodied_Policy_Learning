import utils
import torch
from tqdm import tqdm
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from evaluate import load
bertscore = load('bertscore')
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
import numpy as np
from schema import *
from preprocess_utils import *
import sys
sys.path.append('virtualhome1')
sys.path.append('virtualhome1/simulation')
sys.path.append('../BLIP')
from virtualhome1.simulation.evolving_graph.scripts import read_script_from_list_string, read_script_from_string
from virtualhome1.simulation.evolving_graph.utils import graph_dict_helper
from virtualhome1.dataset_utils.execute_script_utils import *
import virtualhome1.dataset_utils.add_preconds as add_preconds
import virtualhome1.simulation.evolving_graph.check_programs as check_programs
from virtualhome1.simulation.unity_simulator.comm_unity import UnityCommunication

def train(model, image_caption_model, data_loader, optimizer, device):
    model.train()      
    losses= []
    for batch in tqdm(data_loader, desc = 'Training: '):
        optimizer.zero_grad()
        caption, image, _, exec, _, _ = batch

        image = image.to(device)   

        decoder_out, _ = image_caption_model(image, caption)
        del image
        del caption
        decoder_out = torch.mean(decoder_out, dim=1)
        decoder_out = decoder_out.unsqueeze(1)
        exec_input_ids = exec.squeeze(1)
        _, loss = model(decoder_out, exec_input_ids)
        del _              
        loss.backward()
        optimizer.step()    
        
        losses.append(loss.cpu().detach().numpy())
    return average(losses)

def val(model, image_caption_model, data_loader, device):
    model.eval()  
    
    losses = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc = 'Validating: '):
            caption, image, _, exec, _, _ = batch

            image = image.to(device)
            
            decoder_out, _ = image_caption_model(image, caption)
            del image
            del caption
            decoder_out = torch.mean(decoder_out, dim=1)
            decoder_out = decoder_out.unsqueeze(1)
            exec_input_ids = exec.squeeze(1)
            _, loss = model(decoder_out, exec_input_ids)
            del _
            
            losses.append(loss.cpu().detach().numpy())

    return average(losses)  


def test(model, image_caption_model, data_loader, device, tokenizer):
    model.eval() 
    generated_execs = []
    ground_truth_execs = []
    result_1 = []
    ground_truth_1 = []
    result_2 = []
    ground_truth_2 = []
    result_3 = []
    ground_truth_3 = []
    result_4 = []
    ground_truth_4 = []
    result_5 = []
    ground_truth_5 = []
    result_6 = []
    ground_truth_6 = []
    result_7 = []
    ground_truth_7= []
    # rouge_recall = []
    # rouge_precision = []
    # rouge_f1 = []
    meteors=[]
    ids_1 = []
    ids_2 = []
    ids_3 = []
    ids_4 = []
    ids_5 = []
    ids_6 = []
    ids_7 = []
    all_results = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc = 'Testing: '):
            caption, image, id, exec, ground_execs, init_graphs = batch
            image = image.to(device)
            decoder_out, _ = image_caption_model(image, caption)
            del image
            del caption
            decoder_out = torch.mean(decoder_out, dim=1)
            decoder_out = decoder_out.unsqueeze(1)
            exec_input_ids = exec.squeeze(1)
            decoder_hidden, _ = model(decoder_out, exec_input_ids)
            del _
            output_ids = model.llm_generate(inputs_embeds = decoder_hidden)
            gen_execs = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
            print(gen_execs)
            for gen_exec, ground_exec, img_id, graph in zip(gen_execs, ground_execs, id, init_graphs):
                all_results.append({"image_id": img_id, "caption": gen_exec, 'ground_truth' : ground_exec, 'graph' : graph})
                if 'TrimmedTestScene1_graph' in graph:
                    result_1.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_1.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_1.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)
                elif 'TrimmedTestScene2_graph' in graph:
                    result_2.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_2.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_2.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)
                elif 'TrimmedTestScene3_graph' in graph: 
                    result_3.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_3.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_3.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)
                elif 'TrimmedTestScene4_graph' in graph: 
                    result_4.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_4.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_4.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)
                elif 'TrimmedTestScene5_graph' in graph: 
                    result_5.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_5.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_5.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)
                elif 'TrimmedTestScene6_graph' in graph: 
                    result_6.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_6.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_6.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)
                else:
                    result_7.append({"image_id": img_id, "caption": gen_exec})
                    ground_truth_7.append({"image_id": img_id, "caption": ground_exec, 'id' : count})
                    ids_7.append({'id' : img_id})
                    generated_execs.append(gen_exec)
                    ground_truth_execs.append(ground_exec)   
                count +=1
                try: 
                    # r, p, f = calc_rouge(gen_exec, ground_exec)
                    meteor = single_meteor_score(ground_exec.split(), gen_exec.split())
                    meteors.append(meteor)
                    # rouge_recall.append(r)
                    # rouge_precision.append(p)
                    # rouge_f1.append(f)
                except ValueError:
                    pass

    bscore = bertscore.compute(predictions = generated_execs, references = ground_truth_execs, lang='en')
    print(f"Bert Score R is: {average(bscore['recall'])}")
    print(f"Bert Score P is: {average(bscore['precision'])}")
    print(f"Bert Score F is: {average(bscore['f1'])}")
    # print(f'Rouge R is: {average(rouge_recall)}')
    # print(f'Rouge P is: {average(rouge_precision)}')
    # print(f'Rouge F is: {average(rouge_f1)}')
    print(f"Bleu-1 Score is: {calc_bleu_n(ground_truth_execs, generated_execs, 1)}")
    print(f"Bleu-2 Score is: {calc_bleu_n(ground_truth_execs, generated_execs, 2)}")
    print(f"Bleu-3 Score is: {calc_bleu_n(ground_truth_execs, generated_execs, 3)}")
    print(f"Bleu-4 Score is: {calc_bleu_n(ground_truth_execs, generated_execs, 4)}")
    print(f'Meteor Score is: {average(meteors)}')
    return all_results, result_1, ground_truth_1,result_2, ground_truth_2,result_3, ground_truth_3,result_4, ground_truth_4,result_5, ground_truth_5,result_6, ground_truth_6,result_7, ground_truth_7, ids_1, ids_2, ids_3, ids_4, ids_5, ids_6, ids_7
    


def calc_bleu_n(pred, label, n):
    if n == 1:
        score = nltk.translate.bleu_score.corpus_bleu(label, pred, weights=(1, 0, 0, 0))
    elif n == 2:
        score = nltk.translate.bleu_score.corpus_bleu(label, pred, weights = (0.5, 0.5, 0, 0))
    elif n == 3:
        score = nltk.translate.bleu_score.corpus_bleu(label, pred, weights = (0.33, 0.33, 0.33, 0))
    elif n == 4:
        score =nltk.translate.bleu_score.corpus_bleu(label, pred, weights = (0.25, 0.25, 0.25, 0.25))
    return score

def calc_rouge(pred, label):
    rouge = Rouge()
    score = rouge.get_scores(pred, label)
    score = score[0]['rouge-1']
    r = score['r']
    p = score['p']
    f = score['f']
    return r, p, f

def average(lst):
    return sum(lst) / len(lst)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def simulate(results_path):
    results = open_file(results_path)
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
            if result['success_exec'] == 'True':
                successes +=1
                total+=1

                if 'TrimmedTestScene1_graph' in graph:
                    env_1_success +=1
                elif 'TrimmedTestScene2_graph' in graph:
                    env_2_success +=1
                elif 'TrimmedTestScene3_graph' in graph:
                    env_3_success +=1
                elif 'TrimmedTestScene4_graph' in graph:
                    env_4_success +=1
                elif 'TrimmedTestScene5_graph' in graph:
                    env_5_success +=1
                elif 'TrimmedTestScene6_graph' in graph:
                    env_6_success +=1
                else: 
                    env_7_success +=1
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


def finetune_train(model, data_loader, optimizer, device):

    model.train()  
    losses= []
    for batch in tqdm(data_loader, desc = 'Training: '):
        caption, image, _, _, _ = batch

        image = image.to(device)       
        
        _, loss = model(image, caption)
        del _
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        losses.append(loss.cpu().detach().numpy())
    # gather the stats from all processes
    return average(losses)

def finetune_val(model, data_loader, device):
   
    model.eval()  
    losses = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc = 'Validating: '):
            caption, image, _, _, _ = batch

            image = image.to(device)       
            
            _, loss = model(image, caption)
            del _

            losses.append(loss.cpu().detach().numpy())

    return average(losses)  

def train_reinforce(model, image_caption_model, data_loader, optimizer, device, env, tokenizer, state):
    model.train()      
    losses= []
    actions = []
    states = []
    log_probs = []
    for batch in tqdm(data_loader, desc = 'Training: '):
        optimizer.zero_grad()
        caption, image, _, exec, _, _ = batch

        image = image.to(device)   

        decoder_out, _ = image_caption_model(image, caption)
        del image
        del caption
        decoder_out = torch.mean(decoder_out, dim=1)
        decoder_out = decoder_out.unsqueeze(1)
        exec_input_ids = exec.squeeze(1)
        decoder_hidden, _, state, log_prob_action = model(decoder_out, exec_input_ids, state)
        output_ids = model.llm_generate(inputs_embeds = decoder_hidden)
        gen_execs = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
        actions.append(gen_execs)
        states.append(state)
        log_probs.append(log_prob_action)

    for i in range(len(actions)):
        state = states[i]
        action = actions[i]
        log = log_probs[i]
        _, reward, _, _ = env.step(action)
        discounted_rewards = []

        for t in range(len(r)):
            Gt = 0 
            pw = 0
            for r_ in reward[t:]:
                Gt = Gt + 0.99**pw * r_
                pw = pw + 1
            discounted_rewards.append(Gt)
        
        discounted_rewards = np.array(discounted_rewards)

        discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32,device=device)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/ (torch.std(discounted_rewards))
        log_prob = torch.stack(log)
        policy_gradient = -log_prob*discounted_rewards
        losses.append(policy_gradient)

        model.zero_grad()
        policy_gradient.sum().backward()
        optimizer.step()
        
    return average(losses)

def val_reinforce(model, image_caption_model, data_loader, device, env, state, tokenizer):
    model.eval()      
    losses= []
    actions = []
    states = []
    log_probs = []
    for batch in tqdm(data_loader, desc = 'Validating: '):
        caption, image, _, exec, _, _ = batch

        image = image.to(device)   

        decoder_out, _ = image_caption_model(image, caption)
        del image
        del caption
        decoder_out = torch.mean(decoder_out, dim=1)
        decoder_out = decoder_out.unsqueeze(1)
        exec_input_ids = exec.squeeze(1)
        decoder_hidden, _, state, log_prob_action = model(decoder_out, exec_input_ids, state)
        output_ids = model.llm_generate(inputs_embeds = decoder_hidden)
        gen_execs = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
        actions.append(gen_execs)
        states.append(state)
        log_probs.append(log_prob_action)

    for i in range(len(actions)):
        state = states[i]
        action = actions[i]
        log = log_probs[i]
        _, reward, _, _ = env.step(action)
        discounted_rewards = []

        for t in range(len(r)):
            Gt = 0 
            pw = 0
            for r_ in reward[t:]:
                Gt = Gt + 0.99**pw * r_
                pw = pw + 1
            discounted_rewards.append(Gt)
        
        discounted_rewards = np.array(discounted_rewards)

        discounted_rewards = torch.tensor(discounted_rewards,dtype=torch.float32,device=device)
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards))/ (torch.std(discounted_rewards))
        log_prob = torch.stack(log)
        policy_gradient = -log_prob*discounted_rewards
        losses.append(policy_gradient)

        
    return average(losses)