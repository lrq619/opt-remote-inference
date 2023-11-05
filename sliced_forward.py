from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, LogitsProcessorList
from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
import torch.distributed.rpc as rpc
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from preprocess import preprocess_alpaca
from model_loading import baseline_model_loading, remote_model_loading
import time
import torch
import numpy as np
import argparse
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

prompt_num = 16
prompts, gt_responses = preprocess_alpaca(prompt_num)

batch_size = 16
num_batches = (prompt_num + batch_size - 1) // batch_size


# Tokenize batch of prompts and gt_responses
batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

input_ids = batch_inputs["input_ids"]

logits_processor = LogitsProcessorList()

with torch.no_grad():
    past_key_values = None
    for i in range(2):
        if i == 0:
            model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
        else:
            model_inputs = {
                "past_key_values":None,
                "use_cache": True,
                "input_ids":input_ids[:, -1:],
                "past_key_values_length":input_ids.shape[1] - 1
            }
        start = time.time()

        batch_outputs, metrics = model.forward(**model_inputs)

        inference_latencys, comm_overheads, inter_tensor_sizes = metrics

        with ctp.append_run("forward_case_study") as run:
            run.monitor("inference_latencys", inference_latencys)
            run.monitor("comm_overheads", comm_overheads)
            run.monitor("inter_tensor_sizes", inter_tensor_sizes)
        
        inference_latency = sum(inference_latencys)
        comm_overhead = sum(comm_overheads)
        inter_tensor_size = np.mean(inter_tensor_sizes)

        next_token_logits = batch_outputs.logits[:,-1,:]
        past_key_values = batch_outputs.past_key_values
        # print(f"past_key_values:\n{past_key_values}")
        # print(f"type of past_key_values:{type(past_key_values)}")
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        next_token_ids = torch.argmax(next_tokens_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)
        next_tokens = tokenizer.decode(next_token_ids)
        print(f"generated tokens for {i} forward: {next_tokens}")
        print(f"forward latency for {i} forward: {inference_latency:.1f} ms")

rpc.shutdown()