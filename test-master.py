import os
import torch
import torch.distributed.rpc as rpc
import torch.distributed as dist
import time
import argparse
import ctp

from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM
from my_generate import my_generate
from preprocess import preprocess_alpaca
from config import WORKER_LAYERS_MAP
from utils import create_directory

addr = '0.0.0.0'
port = '29500'
os.environ['MASTER_ADDR'] = addr
os.environ['MASTER_PORT'] = port

print(f"master listening on {addr}:{port}")
world_size = 2
rpc.init_rpc("master", rank = 0, world_size = world_size)
rpc.shutdown()
print("finished")