from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, LogitsProcessorList
from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
import torch.distributed.rpc as rpc
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from preprocess import preprocess_alpaca
from model_loading import baseline_model_loading, remote_model_loading,warm_up_remote
from my_generate import my_generate_with_ctp, my_generate_remote
import time
import torch
import numpy as np
import argparse
import utils
import ctp

device="cuda:0"

parser = argparse.ArgumentParser(description="Remote Decoder Layers")
parser.add_argument('--world-size', type=int, required=True, help="World size")

args = parser.parse_args()
world_size = args.world_size

model : RemoteOPTForCausalLM
config, tokenizer, model = remote_model_loading(MODEL_NAME, world_size)

model.to(device)
model.eval()

prompt_num = 1
prompts, gt_responses = preprocess_alpaca(prompt_num)
print(f"prompts: {prompts}")
batch_size = 1
num_batches = (prompt_num + batch_size - 1) // batch_size


# Tokenize batch of prompts and gt_responses
batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

input_ids = batch_inputs["input_ids"]
warm_up_remote(model, input_ids)
model.clear_kv_cache()

logits_processor = LogitsProcessorList()

run = ctp.append_run("prompt")
for batch_idx in range(num_batches):
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, prompt_num)
    
    batch_prompts = prompts[batch_start:batch_end]
    batch_gt_responses = gt_responses[batch_start:batch_end]

    # Tokenize batch of prompts and gt_responses
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

    size = utils.calculate_object_bytes(batch_inputs)/1024
    print(f"prompt size: {size:.1f}KB")
    run.collect("sizes", size)
    batch_gt_outputs = tokenizer(batch_gt_responses, return_tensors="pt", padding=True)
    
    batch_input_token_lengths = [len(input_ids) for input_ids in batch_inputs["input_ids"]]
    print(f"input token_length: {batch_input_token_lengths}")
    batch_gt_output_token_lengths = [len(output_ids) for output_ids in batch_gt_outputs["input_ids"]]
    print(f"output_token_length: {batch_gt_output_token_lengths}")

    # Note: In the batched approach, we use the max token length in the batch for generation
    max_gt_output_token_length = max(batch_gt_output_token_lengths)

    with torch.no_grad():
        batch_generate_ids,_ = my_generate_remote(model, batch_inputs["input_ids"], max_new_tokens=max_gt_output_token_length)

    print(f"batch_generate_ids: {batch_generate_ids}")
    batch_output_tokens = utils.get_output_token_from_generate(batch_generate_ids, tokenizer)
    print(f"batch_output_tokens: {batch_output_tokens}")
    size = utils.calculate_object_bytes(batch_output_tokens)
    print(f"output_token size: {size}B")

rpc.shutdown()