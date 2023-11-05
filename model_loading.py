from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, OPTConfig, PreTrainedTokenizer
from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
import os
import torch.distributed.rpc as rpc
from utils import create_directory
from config import WORKER_LAYERS_MAP
import torch


def baseline_model_loading(model_name:str) -> (OPTConfig, PreTrainedTokenizer,OPTForCausalLM):
    config = AutoConfig.from_pretrained(f"facebook/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}",padding_side='left')
    model = OPTForCausalLM.from_pretrained(f"facebook/{model_name}").half()
    return config, tokenizer, model

def remote_model_loading(model_name:str, world_size:int) -> (OPTConfig, PreTrainedTokenizer, RemoteOPTForCausalLM):

    master_addr = "0.0.0.0"
    master_port = '29500'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"master listening on {master_addr}:{master_port}")
    rpc.init_rpc("master", rank = 0, world_size = world_size)
    print("master init finished")

    print("start model loading")
    config = AutoConfig.from_pretrained(f"facebook/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}",padding_side='left')
    whole_model = OPTForCausalLM.from_pretrained(f"facebook/{MODEL_NAME}").half()
    print("model loading finished")

    # save layer's state dict
    create_directory(f"{STATE_DICT_PATH}")
    create_directory(f"{STATE_DICT_PATH}/{MODEL_NAME}")
    for worker in WORKER_LAYERS_MAP.keys():
        layers_range = WORKER_LAYERS_MAP[worker]
        layers = whole_model.get_decoder().layers[layers_range[0]:layers_range[-1]+1]
        save_dict_path = f"{STATE_DICT_PATH}/{MODEL_NAME}/layers-{layers_range[0]}-{layers_range[-1]}.pth"
        if os.path.isfile(save_dict_path):
            print(f"{save_dict_path} already stored, skip saving")
        else:
            torch.save(layers.state_dict(), save_dict_path)
            print(f"saved layers to {save_dict_path}")

    model = RemoteOPTForCausalLM(config, worker_layer_map=WORKER_LAYERS_MAP)

    del model.get_decoder().layers

    embed_tokens = whole_model.get_decoder().embed_tokens
    embed_positions = whole_model.get_decoder().embed_positions
    final_layer_norm = whole_model.get_decoder().final_layer_norm
    lm_head = whole_model.lm_head

    model.set_embeddings(embed_tokens, embed_positions)
    model.set_final_norm(final_layer_norm, lm_head)

    del whole_model

    return config, tokenizer, model