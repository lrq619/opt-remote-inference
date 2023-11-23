from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, LogitsProcessorList
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
from preprocess import preprocess_alpaca
from model_loading import baseline_model_loading, warm_up,warm_up_remote, remote_model_loading
import time
import torch
import ctp
import argparse
from my_generate import my_generate, my_generate_whole_model, my_generate_remote
import torch.distributed.rpc as rpc

device="cuda:0"

parser = argparse.ArgumentParser(description="Remote Decoder Layers")
parser.add_argument('--world-size', type=int, required=True, help="World size")

args = parser.parse_args()
world_size = args.world_size

model : RemoteOPTForCausalLM
config, tokenizer, model = remote_model_loading(MODEL_NAME, world_size)

model.to(device)
model.eval()

prompt_num = 1024
prompts, gt_responses = preprocess_alpaca(prompt_num)
print(f"length of prompts: {len(prompts)}, length of responses: {len(gt_responses)}")

batch_size = 16
num_batches = (prompt_num + batch_size - 1) // batch_size

# batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

# input_ids = batch_inputs["input_ids"]


run = ctp.append_run("sliced_model_serving")
for batch_idx in range(num_batches):
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, prompt_num)
    
    batch_prompts = prompts[batch_start:batch_end]
    batch_gt_responses = gt_responses[batch_start:batch_end]

    # Tokenize batch of prompts and gt_responses
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
    if batch_idx == 0:
        warm_up_remote(model, batch_inputs["input_ids"])
        model.clear_kv_cache()
    batch_gt_outputs = tokenizer(batch_gt_responses, return_tensors="pt", padding=True)
    
    batch_input_token_lengths = [len(input_ids) for input_ids in batch_inputs["input_ids"]]
    batch_gt_output_token_lengths = [len(output_ids) for output_ids in batch_gt_outputs["input_ids"]]

    # Note: In the batched approach, we use the max token length in the batch for generation
    max_gt_output_token_length = max(batch_gt_output_token_lengths)

    with torch.no_grad():
        start = time.time()
        batch_generate_ids,metrics = my_generate_remote(model, batch_inputs["input_ids"], max_new_tokens=max_gt_output_token_length)
        inference_latency, comm_overhead, serving_latency = metrics

        run.collect('serving_latencys', serving_latency)
        run.collect('inference_latencys', inference_latency)
        run.collect('comm_overheads', comm_overhead)
    del batch_inputs

run.stop_collect()
rpc.shutdown()