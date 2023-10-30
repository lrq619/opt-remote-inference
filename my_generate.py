from transformers import pipeline, OPTConfig, OPTModel,AutoTokenizer, OPTForCausalLM, LogitsProcessorList
from preprocess import preprocess_alpaca
import time
import copy
import torch
import time

def my_generate(model : OPTForCausalLM, input_ids, max_new_tokens):
    _input_ids = copy.copy(input_ids)
    logits_processor = LogitsProcessorList()
    past_key_values = None
    for i in range(max_new_tokens):
        past_key_values = None
        model_inputs = model.prepare_inputs_for_generation(input_ids=_input_ids, use_cache=True, past_key_values=past_key_values)
        outputs, inference_latency, comm_overhead = model.forward(**model_inputs)

        next_token_logits = outputs.logits[:,-1,:]
        past_key_values = outputs.past_key_values
        next_tokens_scores = logits_processor(_input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        _input_ids = torch.cat([_input_ids, next_tokens[:, None]], dim=-1)

    return _input_ids
