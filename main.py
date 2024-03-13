import os
import json
import torch
from torch.utils.data import DataLoader
from models.blip import blip_decoder
from utils import cosine_lr_schedule
from data.utils import save_result, coco_caption_eval
import pandas as pd
from dataset import *
import schema
from train_utils import *
import matplotlib.pyplot as plt
from transformers import OFAModel, OFATokenizer, BertTokenizer, BertModel, EncoderDecoderModel, BartModel, BartTokenizer, RobertaModel, RobertaTokenizer
from models import APM
from GRIT.models.caption import Transformer, GridFeatureNetwork, CaptionGenerator
from GRIT.models.caption.detector import build_detector
from GRIT.models.common.attention import MemoryAttention
from OFA_hf.models.ofa.ofa import OFAModel
import gym

import gc
import yaml
import argparse



def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--sum', type=str, default = 'blip', help="Please choose a SUM from the following list: ['grit', 'bip', 'ofa']")
    parser.add_argument('--apm', type=str, default = 'bert', help = "Please choose a APM from the following list: ['bert', 'bart', 'roberta']")
    parser.add_argument('--finetunesum', type = bool, default = False, help = "Please choose whether to finetune the sum model or not")
    parser.add_argument('--finetuneapm', type = bool, default = False, help = "Please choose whether to finetune the apm model or not")
    parser.add_argument('--inference', type = bool, default = True, help = "Please choose whether to run inference or not")
    parser.add_argument('--learning', type = str, default = 'supervised', help="Please choose the learning paradigm between the following: ['supervised', 'reinforce']")
    
    return parser.parse_args()

if __name__ == '__main__':
    
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args = get_args()


    if args.apm == 'bert':
        llm_tokenizer = BertTokenizer.from_pretrained(schema.pretrained_embed_model_name)
        llm = BertModel.from_pretrained(schema.pretrained_embed_model_name, output_hidden_states = True)
    elif args.apm == 'bart':
        llm_tokenizer = BartTokenizer.from_pretrained(schema.pretrained_embed_model_name)
        llm = BartModel.from_pretrained(schema.pretrained_embed_model_name, output_hidden_states = True)
    else:
        llm_tokenizer = RobertaTokenizer.from_pretrained(schema.pretrained_embed_model_name)
        llm = RobertaModel.from_pretrained(schema.pretrained_embed_model_name, output_hidden_states = True)

    llm = EncoderDecoderModel.from_encoder_decoder_pretrained(schema.pretrained_embed_model_name, schema.pretrained_embed_model_name, output_hidden_states = True)
    llm.config.decoder_start_token_id = llm_tokenizer.cls_token_id
    llm.config.pad_token_id = llm_tokenizer.pad_token_id
    llm.config.is_encoder_decoder=True

    train_tsv = pd.read_csv(schema.train_path, sep='\t').values
    val_tsv = pd.read_csv(schema.val_path, sep='\t').values
    test_tsv = pd.read_csv(schema.test_path, sep='\t').values

    train_ds = ImageCaptionDataset(
        caption = train_tsv[:, 2],
        img_paths = train_tsv[:, 1],
        resolution = schema.resolution,
        mean = schema.mean,
        std = schema.std,
        device = device,
        init_graphs = train_tsv[:, 6],
        id = train_tsv[:, 0],
        tokenizer = llm_tokenizer,
        embed_model = llm,
        exec = train_tsv[:, 5],
        max_len=schema.max_length
    )
    val_ds = ImageCaptionDataset(
        caption = val_tsv[:, 2],
        img_paths = val_tsv[:, 1],
        resolution = schema.resolution,
        mean = schema.mean,
        std = schema.std,
        device = device,
        init_graphs = val_tsv[:, 6],
        id = val_tsv[:, 0],
        tokenizer = llm_tokenizer,
        embed_model = llm,
        exec = val_tsv[:, 5],
        max_len=schema.max_length
    )


    train_loader = DataLoader(
        dataset = train_ds,
        batch_size = schema.batch_size,
        shuffle= True
    )
    val_loader = DataLoader(
        dataset = val_ds,
        batch_size = schema.batch_size,
        shuffle= True
    )

    if args.sum == 'blip':
        sum = blip_decoder(pretrained = schema.pretrained, image_size = schema.resolution, vit = schema.vit, 
                                vit_grad_ckpt = schema.vit_grad_ckpt, vit_ckpt_layer = schema.vit_ckpt_layer,
                                prompt = schema.prompt).to(device)
    elif args.apsumm == 'ofa':
        sum = OFAModel.from_pretrained(schema.pretrained)
    else:
        with open('GRIT/configs/caption/coco_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        detector = build_detector(config)

        grit_net = GridFeatureNetwork(
            pad_idx=config.model.pad_idx,
            d_in=config.model.grid_feat_dim,
            dropout=config.model.dropout,
            attn_dropout=config.model.attn_dropout,
            attention_module=MemoryAttention,
            **config.model.grit_net,
        )
        cap_generator = CaptionGenerator(
            vocab_size=config.model.vocab_size,
            max_len=config.model.max_len,
            pad_idx=config.model.pad_idx,
            dropout=config.model.dropout,
            attn_dropout=config.model.attn_dropout,
            cfg=config.model.cap_generator,
            **config.model.cap_generator,
        )
        sum = Transformer(
            grit_net,
            cap_generator,
            detector=detector, # .module,
            use_gri_feat=config.model.use_gri_feat,
            use_reg_feat=config.model.use_reg_feat,
            config=config,
        )
    
    if args.finetunesum == False:
        finetuned_model_name = f'{schema.output_dir}/finetuned_checkpoint_best_{args.sum}_{args.apm}.pth'
        finetuned_chkpt = torch.load(finetuned_model_name, map_location = 'cuda')
        sum.load_state_dict(finetuned_chkpt['model'])
    else:
        pass
    apm = APM(llm, args).to(device)

    # chkpt = torch.load(os.path.join(schema.output_dir, f'checkpoint_best_{schema.vit}_{schema.vit_ckpt_layer}_{schema.batch_size}_{schema.init_lr}_{schema.max_length}_{schema.num_beams}_{schema.weight_decay}.pth'), map_location='cuda')
    # model.load_state_dict(chkpt['model'])

    optimizer = torch.optim.AdamW(params = llm.parameters(), lr = schema.init_lr, weight_decay=schema.weight_decay)

    early_stopping = EarlyStopper(patience=5, min_delta=0.2)

    best = 0
    best_epoch = 0
    val_losses = []
    train_losses = []
    all_epochs = []
    if args.learning == 'reinforce':
        env = gym.make()

    for epoch in range(0, schema.max_epoch):

        cosine_lr_schedule(optimizer, epoch, schema.max_epoch, schema.init_lr, schema.min_lr)
        
        if args.finetunesum == True:
            train_loss = finetune_train(sum, train_loader, optimizer, device)
            val_loss = finetune_val(sum, val_loader, device)
        else:
            if args.learning == 'supervised':
                train_loss = train(apm, sum, train_loader, optimizer, device)
                val_loss = val(apm, sum, val_loader, device)
            else:
                state = env.reset()
                train_loss = train_reinforce(apm, sum, train_loader, optimizer, device, env, llm_tokenizer, state)
                state = env.reset()
                val_loss = val_reinforce(apm, sum, val_loader, optimizer, device, env, llm_tokenizer, state)

        print(f'Train Loss: {train_loss}')
        train_losses.append(train_loss)
        print(f'Val Loss: {val_loss}')
        val_losses.append(val_loss)

        model_state_dict = apm.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'config_file': 'config',
            'epoch': epoch}

        if val_loss <= min(val_losses):
            torch.save(checkpoint, os.path.join(schema.output_dir, f'checkpoint_best_{args.sum}_{args.apm}.pth')) 
            print('    - [Info] The checkpoint file has been updated.')

        all_epochs.append(epoch)

        if early_stopping.early_stop(val_loss):
            print("We are at epoch:", epoch)
            break


    print('ALL DONE')               
    fig1 = plt.figure('Figure 1')
    plt.plot(train_losses, label = 'train')
    plt.plot(val_losses, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, 4])
    plt.ylabel('loss')
    plt.legend(loc ="upper right")
    plt.title('loss change curve')
    plt.savefig(f'pngs/checkpoint_best_{args.sum}_{args.apm}.png')

    if args.inference == True:

        test_ds = ImageCaptionDataset(
        caption = test_tsv[:, 2],
        img_paths = test_tsv[:, 1],
        resolution = schema.resolution,
        mean = schema.mean,
        std = schema.std,
        device = device,
        init_graphs = test_tsv[:, 6],
        id = test_tsv[:, 0],
        tokenizer = llm_tokenizer,
        embed_model = llm,
        exec = test_tsv[:, 5],
        max_len=schema.max_length
    )

        test_loader = DataLoader(
        dataset = test_ds,
        batch_size = schema.batch_size,
        shuffle= True
    )
        chkpt = torch.load(os.path.join(schema.output_dir, f'checkpoint_best_{args.sum}_{args.apm}.pth'), map_location='cuda')
        apm.load_state_dict(chkpt['model'])

        all_results, result_1, ground_truth_1,result_2, ground_truth_2,result_3, ground_truth_3,result_4, ground_truth_4,result_5, ground_truth_5,result_6, ground_truth_6,result_7, ground_truth_7, ids_1, ids_2, ids_3, ids_4, ids_5, ids_6, ids_7 = test(apm, sum, test_loader, device, llm_tokenizer)

        all_result_file = save_result(all_results, schema.result_dir, 'all_results', remove_duplicate='image_id')

        test_result_file = save_result(result_1, schema.result_dir, 'test_results1', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_1, schema.ground_truth_dir, 'ground_results1', remove_duplicate='image_id', ids = ids_1)        
        print('calculating env 1')
        coco_test = coco_caption_eval(schema.ground_result_file_path,schema.test_result_file_path)

        test_result_file = save_result(result_2, schema.result_dir, 'test_results2', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_2, schema.ground_truth_dir, 'ground_results2', remove_duplicate='image_id', ids = ids_2)        
        print('calculating env 2')
        coco_test = coco_caption_eval(schema.ground_result_file_path2,schema.test_result_file_path2)

        test_result_file = save_result(result_3, schema.result_dir, 'test_results3', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_3, schema.ground_truth_dir, 'ground_results3', remove_duplicate='image_id', ids = ids_3)        
        print('calculating env 3')
        coco_test = coco_caption_eval(schema.ground_result_file_path3,schema.test_result_file_path3)

        test_result_file = save_result(result_4, schema.result_dir, 'test_results4', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_4, schema.ground_truth_dir, 'ground_results4', remove_duplicate='image_id', ids = ids_4)        
        print('calculating env 4')
        coco_test = coco_caption_eval(schema.ground_result_file_path4,schema.test_result_file_path4)

        test_result_file = save_result(result_5, schema.result_dir, 'test_results5', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_5, schema.ground_truth_dir, 'ground_results5', remove_duplicate='image_id', ids = ids_5)        
        print('calculating env 5')
        coco_test = coco_caption_eval(schema.ground_result_file_path5,schema.test_result_file_path5)

        test_result_file = save_result(result_6, schema.result_dir, 'test_results6', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_6, schema.ground_truth_dir, 'ground_results6', remove_duplicate='image_id', ids = ids_6)        
        print('calculating env 6')
        coco_test = coco_caption_eval(schema.ground_result_file_path6,schema.test_result_file_path6)

        test_result_file = save_result(result_7, schema.result_dir, 'test_results7', remove_duplicate='image_id')        
        ground_result_file = save_result(ground_truth_7, schema.ground_truth_dir, 'ground_results7', remove_duplicate='image_id', ids = ids_7)        
        print('calculating env 7')
        coco_test = coco_caption_eval(schema.ground_result_file_path7,schema.test_result_file_path7)

        results_path = 'results/all_results.json'
        simulate(results_path)
        # log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}
        #             }
        # with open(os.path.join(schema.output_dir, "evaluate.txt"),"a") as f:
        #     f.write(json.dumps(log_stats) + "\n")

