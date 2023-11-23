from transformers import pipeline, OPTConfig, OPTModel,AutoTokenizer, OPTForCausalLM, LogitsProcessorList
from preprocess import preprocess_alpaca
import time
import copy
import torch
import time
from data_process import forward_case_study, generate_case_study
import numpy as np
import ctp

def process_metrics(metrics):
    # forward_case_study(metrics) 
    generate_case_study(metrics)


def my_generate(model : OPTForCausalLM, input_ids, max_new_tokens):
    _input_ids = copy.copy(input_ids)
    logits_processor = LogitsProcessorList()
    past_key_values = None
    inference_latency = 0
    comm_overhead = 0
    e2e_latency = 0
    for i in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids=_input_ids, use_cache=True, past_key_values=past_key_values)
        outputs = model.forward(**model_inputs)

        # inference_latencys, comm_overheads, inter_tensor_sizes = metrics
        # inference_latency += sum(inference_latencys)
        # comm_overhead += sum(comm_overheads)
        # e2e_latency += sum(inference_latencys) + sum(comm_overheads)
        # process_metrics(metrics)
        

        next_token_logits = outputs.logits[:,-1,:]
        past_key_values = outputs.past_key_values
        next_tokens_scores = logits_processor(_input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        _input_ids = torch.cat([_input_ids, next_tokens[:, None]], dim=-1)

    return _input_ids, inference_latency, comm_overhead, e2e_latency

def my_generate_remote(model : OPTForCausalLM, input_ids, max_new_tokens):
    model.clear_kv_cache()
    _input_ids = copy.copy(input_ids)
    logits_processor = LogitsProcessorList()
    past_key_values = None
    inference_latency = 0
    comm_overhead = 0
    serving_latency = 0
    for i in range(max_new_tokens):
        if i == 0:
            model_inputs = model.prepare_inputs_for_generation(input_ids=_input_ids, use_cache=True, past_key_values=past_key_values)
        else:
            model_inputs = {
                "past_key_values":None,
                "use_cache": True,
                "input_ids":_input_ids[:, -1:],
                "past_key_values_length":_input_ids.shape[1] - 1
            }
        outputs, metrics = model.forward(**model_inputs)

        inference_latencys, comm_overheads, inter_tensor_sizes = metrics
        inference_latency += sum(inference_latencys)
        comm_overhead += sum(comm_overheads)
        serving_latency += sum(inference_latencys) + sum(comm_overheads)
        

        next_token_logits = outputs.logits[:,-1,:]
        past_key_values = outputs.past_key_values
        next_tokens_scores = logits_processor(_input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        _input_ids = torch.cat([_input_ids, next_tokens[:, None]], dim=-1)

    return _input_ids, (inference_latency, comm_overhead, serving_latency)


def my_generate_with_ctp(model : OPTForCausalLM, input_ids, max_new_tokens):
    model.clear_kv_cache()
    _input_ids = copy.copy(input_ids)
    logits_processor = LogitsProcessorList()
    past_key_values = None
    run = ctp.append_run("generate_case_study")
    for i in range(max_new_tokens):
        if i == 0:
            model_inputs = model.prepare_inputs_for_generation(input_ids=_input_ids, use_cache=True, past_key_values=past_key_values)
        else:
            model_inputs = {
                "past_key_values":None,
                "use_cache": True,
                "input_ids":_input_ids[:, -1:],
                "past_key_values_length":_input_ids.shape[1] - 1
            }
        # model_inputs = model.prepare_inputs_for_generation(input_ids=_input_ids, use_cache=True, past_key_values=past_key_values)
        outputs, metrics = model.forward(**model_inputs)

        inference_latencys, comm_overheads, inter_tensor_sizes = metrics
        run.collect("inference_latencys", sum(inference_latencys))
        run.collect("comm_overheads", sum(comm_overheads))
        run.collect("inter_tensor_sizes", sum(inter_tensor_sizes))
        # process_metrics(metrics)

        next_token_logits = outputs.logits[:,-1,:]
        past_key_values = outputs.past_key_values
        next_tokens_scores = logits_processor(_input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        _input_ids = torch.cat([_input_ids, next_tokens[:, None]], dim=-1)
    run.stop_collect()

    return _input_ids

def my_generate_whole_model(model : OPTForCausalLM, input_ids, max_new_tokens):
    _input_ids = copy.copy(input_ids)
    logits_processor = LogitsProcessorList()
    past_key_values = None
    run = ctp.append_run("transformer_generate_case_study")
    for i in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids=_input_ids, use_cache=True, past_key_values=past_key_values)
        start = time.time()
        outputs = model.forward(**model_inputs)
        inference_latency = 1000*(time.time() - start)

        run.collect("inference_latencys", inference_latency)
        # process_metrics(metrics)

        next_token_logits = outputs.logits[:,-1,:]
        past_key_values = outputs.past_key_values
        next_tokens_scores = logits_processor(_input_ids, next_token_logits)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        _input_ids = torch.cat([_input_ids, next_tokens[:, None]], dim=-1)
    run.stop_collect()

    return _input_ids
