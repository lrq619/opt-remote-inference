import os
import torch
import torch.distributed.rpc as rpc
import torch.distributed as dist
from remote_opt.remote_opt_for_causal_lm import RemoteOPTForCausalLM
from remote_opt.config import MODEL_NAME
from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM
from my_generate import my_generate
from preprocess import preprocess_alpaca

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
# dist.init_process_group(backend='gloo', rank=0, world_size=2)
rpc.init_rpc("worker0", rank = 0, world_size = 2)

worker_layer_map = {
    1:range(0,32)
}
device = "cuda:0"

config = AutoConfig.from_pretrained(f"facebook/{MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{MODEL_NAME}")
whole_model = OPTForCausalLM.from_pretrained(f"facebook/{MODEL_NAME}").half()

model = RemoteOPTForCausalLM(config, worker_layer_map=worker_layer_map)
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
prompts, gt_outputs = preprocess_alpaca(prompt_num=prompt_num)
prompt = prompts[0]
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

outputs = model.forward(input_ids)
logits = outputs.logits
output_token_id = torch.argmax(logits[:,-1,:])
output_token = tokenizer.decode(output_token_id)
print(output_token)
prompt += output_token



rpc.shutdown()