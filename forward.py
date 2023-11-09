from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, LogitsProcessorList
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from preprocess import preprocess_alpaca
from model_loading import baseline_model_loading, warm_up
import time
import torch

device="cuda:0"

model:OPTForCausalLM
config, tokenizer, model = baseline_model_loading(MODEL_NAME)

model.to(device)
model.eval()

prompt_num = 16
prompts, gt_responses = preprocess_alpaca(prompt_num)

batch_size = 16
num_batches = (prompt_num + batch_size - 1) // batch_size


# Tokenize batch of prompts and gt_responses
batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

input_ids = batch_inputs["input_ids"]
warm_up(model, input_ids)
print(f"finish warm up")

logits_processor = LogitsProcessorList()

with torch.no_grad():
    past_key_values = None
    for i in range(2):
        # warm_up()
        model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
        start = time.time()

        batch_outputs = model.forward(**model_inputs)

        forward_latency = (time.time() - start) * 1000

        next_token_logits = batch_outputs.logits[:,-1,:]
        past_key_values = batch_outputs.past_key_values
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        next_token_ids = torch.argmax(next_tokens_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)
        next_tokens = tokenizer.decode(next_token_ids)
        print(f"generated tokens for {i} forward: {next_tokens}")
        print(f"forward latency for {i} forward: {forward_latency:.1f} ms")
