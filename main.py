"""
The MIT License

Copyright (c) 2026 Samsung SDS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
DEBUG_MODE_POST_PROCESSING = False
EVALUATION_ONLY = False
ACCELERATOR = 'cuda'


import os
import platform
import torch
from utils import get_device_name

def is_running_on_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

IS_IPYTHON = is_running_on_ipython()        
PLATFORM = platform.system()
NUM_NODES = 1
   
NUM_DEVICES = torch.cuda.device_count()
DEVICE_NAME = 'UNKNOWN'
RANDOM_SEED = 1234

if PLATFORM=='Windows':
    DEBUG_MODE = True


device_full_name = get_device_name().lower()
if 'h100' in device_full_name:
    DEVICE_NAME = 'H100'
elif 'v100' in device_full_name:
    DEVICE_NAME = 'V100'

    
##########################################################################################
# Path Config

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURR_DIR)
DATA_DIR = '/data/dataset'

IS_GPU_C = True if '/home/jovyan' in CURR_DIR else False

if IS_GPU_C:
    DATA_DIR = '/home/jovyan/src/rlft/dataset'
elif PLATFORM=='Windows':
    DATA_DIR = 'D:/data/dataset'


os.chdir(CURR_DIR)



##########################################################################################
# Parameters s

train_data_params = {
    'data_type' : 'bactrianx',
    'data_file' : os.path.join(DATA_DIR, "Bactrian-X-filtered/data/{lang}.json"),    
    'max_gen_tokens': 512,

    'target_language': 'ko',
    'ignore_english': True,

    'debug_batch_len': None,
}

valid_data_params = {
    'data_type' : 'gsm8k',
    'data_file' : os.path.join(DATA_DIR, "gsm8k-platinum/main/test-00000-of-00001.parquet"),
    'max_gen_tokens': 512,

    'target_language': 'ko',
    'ignore_english': True,

    'debug_batch_len': None,
}

model_params = {
    'model_type': 'llama',
    'model_path': os.path.join(DATA_DIR, "pretrained_model/Meta-Llama-3.1-8B-Instruct"),

    #'precision': 'bf16-true',
    #'precision': 'bf16-mixed',
    'precision': '32-true',
}

optimizer_params = {
    'lr': 5e-7,
    'end_lr': 5e-8,
    'lr_decay_per_epoch': 0.9,
    'warmup_rate': 0.1,
    'weight_decay': 0.1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_norm_clip': 1.,
    
    'kld_beta': 0.04,

    'ppo_epsilon_up': 0.2,
    'ppo_epsilon_dn': 0.2,

    'ppo_reuse': 1,
}

train_params = {
    'trainer_type': 'tlpo',
    'num_epoch': 1,          

    # resume
    'start_epoch': 1,
    'step_idx': 0,

    # batch
    'update_batch_size': 128,          # Actual training batch size, this value shold be a multiple of (per_device_update_batch_size*num_of_gpus)
    'per_device_forward_batch_size': 16,  # forward/backward batch size per GPU
    'per_device_gen_batch_size': 32,    # generation batch size per GPU, MUST be multiple of num_concurrent_solutions
    
    # generation
    'num_concurrent_solutions': 8,  
    'num_token_extension': 16,
    
    'temperature': 0.6,
        
    # evaluation
    'eval_before_train': True,
    'eval_num_in_epoch': 20,    

    # save test
    'save_test': True,
    'save_interval': 1,

    # save response csv log
    'save_response_csv': True,
    
}


logger_params = {
    'log_file': {
        'desc': 'v16',
        'filename': 'log.txt'
    }
}


def _set_custom_parameters():
    global train_data_params
    global train_params
    global logger_params
    global optimizer_params

    if train_params['trainer_type']=='grpo':
        train_params['eval_num_in_epoch'] *= 4
    

def _set_debug_mode_parameters():
    global DEBUG_MODE_POST_PROCESSING
    global train_data_params
    global valid_data_params
    global model_params
    global train_params
    global logger_params


    logger_params['log_file']['desc'] = 'debug_' + logger_params['log_file']['desc']

    train_data_params['debug_batch_len'] = 8 
    valid_data_params['debug_batch_len'] = 4

    train_params['save_test'] = False
    train_params['num_epoch'] = 1
    train_params['update_batch_size'] = 4

    train_params['per_device_forward_batch_size'] = 2
    train_params['per_device_gen_batch_size'] = 16
    train_params['num_concurrent_solutions'] = 8
    
    train_data_params['max_gen_tokens'] = 256 
    valid_data_params['max_gen_tokens'] = 128

    train_params['eval_num_in_epoch'] = 1
    train_params['eval_before_train'] = False

    model_params['precision'] = 'bf16-true'

    if model_params['model_type']=='llama':
        model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/Meta-Llama-3.2-3B-Instruct")
       
    #DEBUG_MODE_POST_PROCESSING = True

    
def parse_args(args_str=None):
    global DEBUG_MODE
    global NUM_NODES
    global optimizer_params
    global train_params
    global train_data_params
    global valid_data_params
    global logger_params    
    global model_params
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_nodes', type=int, default=None)    
    parser.add_argument('-debug', action="store_true", default=None)    
    parser.add_argument('-desc', type=str, default=None)

    parser.add_argument('-num_epoch', type=int, default=None)    
    parser.add_argument('-lr', type=float, default=None)    
    parser.add_argument('-end_lr', type=float, default=None)    
    parser.add_argument('-num_warmup_steps', type=int, default=None)    
    parser.add_argument('-ppo_reuse', type=int, default=None)
    parser.add_argument('-kld_beta', type=float, default=None)

    parser.add_argument('-update_batch_size', type=int, default=None)   
    parser.add_argument('-per_device_forward_batch_size', type=int, default=None)   
    parser.add_argument('-per_device_gen_batch_size', type=int, default=None)
    parser.add_argument('-save_response_csv', type=lambda x: x.lower() == 'true', default=None)

    parser.add_argument('-model_type', type=str, default=None)
    parser.add_argument('-trainer_type', type=str, default=None)
    
    parser.add_argument('-precision', type=str, default=None)
    parser.add_argument('-num_concurrent_solutions', type=int, default=None)
    parser.add_argument('-num_token_extension', type=int, default=None)

    parser.add_argument('-target_language', type=str, default=None)
    parser.add_argument('-ignore_english', type=lambda x: x.lower() == 'true', default=None)

    
    
    args = parser.parse_args() if args_str is None else parser.parse_args(args=args_str.split())


    DEBUG_MODE = DEBUG_MODE if args.debug is None else args.debug
    NUM_NODES = NUM_NODES if args.num_nodes is None else args.num_nodes


    if args.desc is not None:
        logger_params['log_file']['desc'] = args.desc        


    if args.model_type is not None:
        model_params['model_type'] = args.model_type
        if model_params['model_type']=='llama':
            model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/Meta-Llama-3.1-8B-Instruct")
        elif model_params['model_type']=='qwen':
            model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/Qwen3-8B")
        elif model_params['model_type']=='phi4-mini':
            model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/Phi-4-mini-instruct")
        elif model_params['model_type']=='gemma4b':
            model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/gemma-3-4b-it")
        elif model_params['model_type']=='gemma12b':
            model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/gemma-3-12b-it")
        elif model_params['model_type']=='ministral':
            model_params['model_path'] = os.path.join(DATA_DIR, "pretrained_model/Ministral-8B-Instruct-2410")
        else:
            raise ValueError("invalid model_type: {}".format(args.model_type))

    if args.trainer_type is not None:
        train_params['trainer_type'] = args.trainer_type


    logger_params['log_file']['desc'] += '_{}'.format(model_params['model_type'])                    
    logger_params['log_file']['desc'] += '_{}'.format(train_params['trainer_type'])


    _set_custom_parameters()  # call after setting trainer_type
       

    if args.ignore_english is not None:
        train_data_params['ignore_english'] = args.ignore_english
        valid_data_params['ignore_english'] = args.ignore_english
        logger_params['log_file']['desc'] += '_ignore_english{}'.format(args.ignore_english)

    if args.precision is not None:
        model_params['precision'] = args.precision
        logger_params['log_file']['desc'] += '_{}'.format(args.precision)
    
    if args.num_epoch is not None:
        train_params['num_epoch'] = args.num_epoch
        logger_params['log_file']['desc'] += '_num_epoch{}'.format(args.num_epoch)        
        
    if args.lr is not None:
        optimizer_params['lr'] = args.lr
        logger_params['log_file']['desc'] += '_lr{}'.format(args.lr)        
        
    if args.end_lr is not None:
        optimizer_params['end_lr'] = args.end_lr
        #logger_params['log_file']['desc'] += '_end_lr{}'.format(args.end_lr)

    if args.kld_beta is not None:
        optimizer_params['kld_beta'] = args.kld_beta
        logger_params['log_file']['desc'] += '_kld_beta{}'.format(args.kld_beta)                

    if args.num_warmup_steps is not None:
        optimizer_params['num_warmup_steps'] = args.num_warmup_steps
        logger_params['log_file']['desc'] += '_num_warmup_steps{}'.format(args.num_warmup_steps)

    if args.ppo_reuse is not None:
        optimizer_params['ppo_reuse'] = args.ppo_reuse
        logger_params['log_file']['desc'] += '_ppo_reuse{}'.format(args.ppo_reuse)                

    if args.update_batch_size is not None:
        train_params['update_batch_size'] = args.update_batch_size

    logger_params['log_file']['desc'] += '_batch_size{}'.format(train_params['update_batch_size'])

    if args.per_device_forward_batch_size is not None:
        train_params['per_device_forward_batch_size'] = args.per_device_forward_batch_size
        logger_params['log_file']['desc'] += '_fbs{}'.format(args.per_device_forward_batch_size)

    if args.per_device_gen_batch_size is not None:
        train_params['per_device_gen_batch_size'] = args.per_device_gen_batch_size
        logger_params['log_file']['desc'] += '_gbs{}'.format(args.per_device_gen_batch_size)

    if args.num_concurrent_solutions is not None:
        train_params['num_concurrent_solutions'] = args.num_concurrent_solutions
        logger_params['log_file']['desc'] += '_num_conc_sol{}'.format(args.num_concurrent_solutions)

    if args.num_token_extension is not None:
        train_params['num_token_extension'] = args.num_token_extension
        logger_params['log_file']['desc'] += '_te{}'.format(args.num_token_extension)

    if args.save_response_csv is not None:
        train_params['save_response_csv'] = args.save_response_csv

    if args.target_language is not None:
        train_data_params['target_language'] = args.target_language
        valid_data_params['target_language'] = args.target_language
        logger_params['log_file']['desc'] += '_{}'.format(args.target_language)
            

def _print_config():
    logger = logging.getLogger()

    def _print_dict(name, params):
        logger.info(f"{name} {{")
        for key, value in params.items():
            logger.info(f"\t{key}: {value},")
        logger.info("}")
        
    logger.info("NUM_NODES: {}".format(NUM_NODES))
    logger.info("NUM_DEVICES: {}".format(NUM_DEVICES))
    logger.info("DEVICE_NAME: {}".format(DEVICE_NAME))
    
    logger.info("DEBUG_MODE: {}".format(DEBUG_MODE))
    logger.info("EVALUATION_ONLY: {}".format(EVALUATION_ONLY))
    logger.info("IS_IPYTHON: {}".format(IS_IPYTHON))
    logger.info("PLATFORM: {}".format(PLATFORM))

    logger.info("ROOT_DIR: {}".format(ROOT_DIR))
    logger.info("CURR_DIR: {}".format(CURR_DIR))
    
    [_print_dict(g_key, globals()[g_key]) for g_key in globals().keys() if g_key.endswith('params')]

    logger.info("result folder: {}".format(get_result_folder()))
    

def _set_seed(fabric, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fabric.seed_everything(seed)  

    


##########################################################################################
# main

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from functools import partial
import argparse
import logging
import traceback
import random
import numpy as np
import shutil
from glob import glob
import pdb

from trainer.trainer_creator import create_trainer
from utils import create_logger, get_result_folder, set_result_folder, copy_src, print_gpuinfo

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

import threading
import time


def gpu_monitor_loop():
    monitor_interval = 60*30 if DEBUG_MODE==False else 60*3
    
    while True:
        print_gpuinfo()
        time.sleep(monitor_interval)


def set_result_folder_all(fabric):
    if fabric.global_rank == 0:
        result_folder = get_result_folder()
        encoded = list(result_folder.encode("utf-8"))
        str_length = len(encoded)
    else:
        str_length = 0

    length_tensor = torch.tensor([str_length], dtype=torch.long)
    length_tensor = fabric.broadcast(length_tensor, src=0)
    total_length = int(length_tensor.item())

    if fabric.global_rank == 0:
        str_tensor = torch.tensor(encoded, dtype=torch.uint8)
    else:
        str_tensor = torch.empty(total_length, dtype=torch.uint8)

    str_tensor = fabric.broadcast(str_tensor, src=0)

    result_folder = bytes(str_tensor.tolist()).decode("utf-8")
    if fabric.global_rank != 0:
        set_result_folder(result_folder)


def main():
    if DEBUG_MODE:
        _set_debug_mode_parameters()


    #########################
    # launch fabric
    if ACCELERATOR == 'cuda':
        if model_params['model_type']=='llama':
            policy_layer = {LlamaDecoderLayer}
        elif model_params['model_type']=='qwen':
            policy_layer = {Qwen3DecoderLayer}
        elif model_params['model_type']=='gemma4b' or model_params['model_type']=='gemma12b':
            policy_layer = {Gemma3DecoderLayer}
        elif model_params['model_type']=='phi4-mini':
            policy_layer = {Phi3DecoderLayer}
        elif model_params['model_type']=='ministral':
            policy_layer = {MistralDecoderLayer}
        else:
            raise ValueError(f"invalid model type: {model_params['model_type']}")

        
        if PLATFORM=='Windows':
            strategy = FSDPStrategy(
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=policy_layer),
                activation_checkpointing_policy=policy_layer,
                cpu_offload=False,
                limit_all_gathers=False,
                process_group_backend='gloo')        
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=policy_layer),
                activation_checkpointing_policy=policy_layer,
                cpu_offload=False,
                limit_all_gathers=False)        

    
        fabric = L.Fabric(
            accelerator=ACCELERATOR,
            num_nodes=NUM_NODES,
            devices=NUM_DEVICES,
            precision=model_params['precision'],
            strategy=strategy)

        fabric.launch()
        print(f'[{fabric.global_rank}/{fabric.world_size}] fabric launched, strategy: {strategy}')

    else:
        fabric = L.Fabric(accelerator="cpu")
        fabric.launch()
        print(f'[{fabric.global_rank}/{fabric.world_size}] fabric launched, accelerator=cpu')


    _set_seed(fabric, RANDOM_SEED+fabric.global_rank)    

    
    #########################
    # Init
    if fabric.global_rank==0:
        create_logger(**logger_params)            

        if DEBUG_MODE==False:
            copy_src(get_result_folder())

        _print_config()

        if DEBUG_MODE==False:
            monitor_thread = threading.Thread(target=gpu_monitor_loop, daemon=True)
            monitor_thread.start()


    fabric.barrier()
    set_result_folder_all(fabric)
    
    
    ####################
    # train model    
    logger = logging.getLogger()
    logger.info("*"*50)
    logger.info(f"trainer: {train_params['trainer_type']}")
    logger.info("*"*50)
    if (IS_GPU_C==True and NUM_DEVICES>1) or DEBUG_MODE==False:        
        try:
            trainer = create_trainer(train_params['trainer_type'],
                              fabric=fabric,
                              train_data_params=train_data_params,
                              valid_data_params=valid_data_params,
                              model_params=model_params,
                              optimizer_params=optimizer_params,
                              train_params=train_params)
            trainer.run()
            
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("An error occurred: %s", e)
    else:
        trainer = create_trainer(train_params['trainer_type'],
                          fabric=fabric,
                          train_data_params=train_data_params,
                          valid_data_params=valid_data_params,
                          model_params=model_params,
                          optimizer_params=optimizer_params,
                          train_params=train_params)
        trainer.run()



    #########################
    # remove checkpoint dirs
    if (PLATFORM=='Windows' or DEBUG_MODE==True) and DEBUG_MODE_POST_PROCESSING==False:
        if fabric.global_rank==0:    
            for match in glob(f'{trainer.model_save_dir}*'):
               print(f"remove checkpoint folders: {match}")
               shutil.rmtree(match)    

    


if __name__ == "__main__":
    parse_args()
    main()
    
