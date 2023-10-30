import os
import torch
import torch.distributed.rpc as rpc
import torch.distributed as dist
import time
import argparse

from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
from remote_opt.config import MODEL_NAME
from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM
from my_generate import my_generate
from preprocess import preprocess_alpaca
from config import WORKER_LAYERS_MAP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

parser = argparse.ArgumentParser(description="Remote Decoder Layers")
parser.add_argument('--world-size', type=int, required=True, help="World size")

args = parser.parse_args()
world_size = args.world_size
rpc.init_rpc("worker0", rank = 0, world_size = world_size)

device = "cuda:0"

config = AutoConfig.from_pretrained(f"facebook/{MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{MODEL_NAME}")
whole_model = OPTForCausalLM.from_pretrained(f"facebook/{MODEL_NAME}").half()

model = RemoteOPTForCausalLM(config, worker_layer_map=WORKER_LAYERS_MAP)
embed_tokens = whole_model.get_decoder().embed_tokens
embed_positions = whole_model.get_decoder().embed_positions
final_layer_norm = whole_model.get_decoder().final_layer_norm
lm_head = whole_model.lm_head

model.set_embeddings(embed_tokens, embed_positions)
model.set_final_norm(final_layer_norm, lm_head)

model = model.to(device)
model.eval()

# start inference
prompt_num = 1
prompts, gt_responses = preprocess_alpaca(prompt_num=prompt_num)

batch_size = 1
num_batches = (prompt_num + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, prompt_num)
    
    batch_prompts = prompts[batch_start:batch_end]
    batch_gt_responses = gt_responses[batch_start:batch_end]

    # Tokenize batch of prompts and gt_responses
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
    batch_gt_outputs = tokenizer(batch_gt_responses, return_tensors="pt", padding=True)
    
    batch_input_token_lengths = [len(input_ids) for input_ids in batch_inputs["input_ids"]]
    batch_gt_output_token_lengths = [len(output_ids) for output_ids in batch_gt_outputs["input_ids"]]

    # Note: In the batched approach, we use the max token length in the batch for generation
    max_gt_output_token_length = max(batch_gt_output_token_lengths)

    start = time.time()
    with torch.no_grad():
        batch_generate_ids = my_generate(model, batch_inputs["input_ids"], max_new_tokens=max_gt_output_token_length)

    generate_latency = (time.time() - start)

    generate_batch_throughput = max_gt_output_token_length * len(batch_prompts) / generate_latency 
    # run.collect(f"batch_inference_latencys", generate_latency)
    # run.collect(f"{batch_size}_batch_throuputs", generate_batch_throughput)

    batch_output_tokens = tokenizer.batch_decode(batch_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(batch_output_tokens)

rpc.shutdown()