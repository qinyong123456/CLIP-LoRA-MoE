
import random
import argparse  
import numpy as np 
import torch

from lora import run_lora

    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    # MoE arguments
    parser.add_argument('--moe_vision_layers', default=0, type=int, help='number of last layers to use MoE for in vision transformer')
    parser.add_argument('--moe_text_layers', default=0, type=int, help='number of last layers to use MoE for in text transformer')
    parser.add_argument('--moe_num_experts', default=4, type=int, help='number of experts in MoE layer')
    parser.add_argument('--moe_top_k', default=2, type=int, help='number of experts to route to for each token')
    parser.add_argument('--moe_dropout', default=0.0, type=float, help='dropout rate for MoE layer')
    parser.add_argument('--train_router', default=False, action='store_true', help='train the MoE router')
    parser.add_argument('--load_balancing_coef', default=0.01, type=float, help='load balancing loss coefficient')

    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args
    

        
