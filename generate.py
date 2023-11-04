
from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, LogitsProcessorList
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from preprocess import preprocess_alpaca
# from my_generate import my_generate_with_ctp, my_generate, my_generate_whole_model
from model_loading import baseline_model_loading
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

    with torch.no_grad():
        batch_generate_ids = model.generate(batch_inputs["input_ids"], max_new_tokens=max_gt_output_token_length)
