import logging
from config.logger_setup import Runtime_Logging
from config.common import *
import time
from schema import *
import torch
from transformers import OFATokenizer, OFAModel
logger = logging.getLogger(__name__)
logSetup = Runtime_Logging()


if __name__ == '__main__':

    conversion_run_time = time.time()
    logSetup.set_log_task("Reading Configuration File")
    logger.info("Reading in details from configuration file")
    config_dir = get_config_path()
    config = load_configuration(config_dir)

    logSetup.set_log_task("Load Hugging Face and Finetuned Model Checkpoints")
    hf_model = torch.load('OFA/hf_checkpoints/pytorch_model.bin')
    ft_model = torch.load('checkpoint_best.pt')['model']
    logger.info(ft_model)
    logger.info(hf_model)
    
    logSetup.set_log_task("Store Parameters as Python Sets")
    hf_keys = set([k for k in hf_model.keys()])
    ft_keys = set([k for k in ft_model.keys()])
    logger.info(len(hf_keys))
    logger.info(len(ft_keys))

    logSetup.set_log_task('Find Common Parameters')
    common_keys = hf_keys.intersection(ft_keys)
    logger.info(len(common_keys))

    logSetup.set_log_task('Align Parameters')
    for k in hf_model.keys():
        if k in common_keys:
            hf_model[k] = ft_model[k]
            del ft_model[k]
            hf_keys.remove(k)
            ft_keys.remove(k)
    logger.info(len(hf_keys))
    logger.info(len(ft_keys))
    logger.info(f'Examples of remaining Hugging Face parameters: {list(hf_keys)[:10]}')
    logger.info(f'Examples of remaining Finetuned OFA parameters: {list(hf_keys)[:10]}')

    logSetup.set_log_task('Align naming of keys')
    for k in ft_model.keys():
        k_pred = k.replace('ffn_layernorm', 'ffn_layer_norm')
        k_pred = k_pred.replace('self_attn_ln', 'self_attn_mid_layer_norm')
        k_pred = k_pred.replace('cross_attn_ln', 'cross_attn_mid_layer_norm')
        k_pred = k_pred.replace('encoder_attn', 'cross_attn')
        if k_pred in hf_keys:
            hf_model[k_pred] = ft_model[k]
            hf_keys.remove(k_pred)
            ft_keys.remove(k)
    logger.info(len(hf_keys))
    logger.info(len(ft_keys))
    
    logSetup.set_log_task('Examine remaining keys')
    logger.info(hf_keys)
    logger.info(ft_keys)

    logSetup.set_log_task('Map Remaining Parameters')
    for k in ft_model.keys():
        k_pred = k.replace('attn_ln', 'self_attn_mid_layer_norm')
        if k_pred in hf_keys:
            hf_model[k_pred] = ft_model[k]
            hf_keys.remove(k_pred)
            ft_keys.remove(k)
    logger.info(len(hf_keys))
    logger.info(len(ft_keys))

    logSetup.set_log_task('Examine remaining keys')
    logger.info(hf_keys)
    logger.info(ft_keys)

    torch.save(hf_model, 'pytorch_model.bin')
    
