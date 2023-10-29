from transformers.models.opt.configuration_opt import OPTConfig 
from transformers.models.opt.modeling_opt import OPTDecoder, OPTDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from typing import Optional, Union, Tuple, List
import pickle
from .config import FORWARD_PORT, MAX_SEND_SIZE, MAX_RECEIVE_SIZE, MODEL_NAME
from torch.distributed import rpc
from .remote_decoder_layers import RemoteOPTDecoderLayers
import time
from .remote_methods import _remote_method
from .utils import get_object_size
from .logger import init_logger



class RemoteOPTDecoder(OPTDecoder):
    def __init__(self, config: OPTConfig, worker_layer_map):
        super().__init__(config)
        self.logger = init_logger(model_name=MODEL_NAME)
        self.worker_layer_map = worker_layer_map
        self.layers_refs = []
        for worker in self.worker_layer_map.keys():
            self.layers_refs.append(rpc.remote(worker, RemoteOPTDecoderLayers, args=(config, self.worker_layer_map[worker])))
    
    # set the embedding layer locally
    def set_embeddings(self, embed_tokens, embed_positions):
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        forward_start = time.time()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        # self.logger.info(f"attention_mask: {attention_mask}")
        # self.logger.info(f"attention_mask.shape: {attention_mask.shape}")
        # self.logger.info(f"input_shape: {input_shape}")
        # self.logger.info(f"inputs_embeds: {inputs_embeds}")
        # self.logger.info(f"past_key_values_length: {past_key_values_length}")
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        # self.logger.info(f"causal_attention_mask:\n{causal_attention_mask}")
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds


        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        embedding_latency = 1000*(time.time() - forward_start)
        self.logger.info(f"embedding latency: {embedding_latency:.1f} ms")

        # decoder_layer_inputs = {}
        # decoder_layer_inputs["hidden_states"] = hidden_states
        # decoder_layer_inputs["attention_mask"] = causal_attention_mask
        # decoder_layer_inputs["layer_head_mask"] = (None)
        # decoder_layer_inputs["output_attentions"] = output_attentions
        # decoder_layer_inputs["use_cache"] = use_cache

        # decoder_layer_inputs_bytes = pickle.dumps(decoder_layer_inputs)
        # next_decoder_cache_bytes = pickle.dumps(next_decoder_cache)
        # past_key_values_bytes = pickle.dumps(past_key_values)

        # transmit through rpc, so first sends to cpu
        hidden_states = hidden_states.to('cpu')
        attention_mask = causal_attention_mask.to('cpu')
        past_key_values = past_key_values.to('cpu') if past_key_values else None

        inference_latencys = []
        comm_overheads = []
        for layers_ref in self.layers_refs:
            inputs = (hidden_states, attention_mask, past_key_values)
            self.logger.info(f"Going to send {get_object_size(inputs)/(1024**2):.1f} MB data")
            start = time.time()
            outputs = _remote_method(RemoteOPTDecoderLayers.forward, layers_ref, hidden_states, attention_mask, past_key_values)
            rtt = time.time() - start
        
            # hidden_states, next_decoder_cache, inference_latency, whole_forward_latency = outputs
            hidden_states = outputs[0] 
            next_decoder_cache += outputs[1]
            inference_latencys.append(outputs[2])

            comm_overhead = rtt - outputs[3]
            comm_overheads.append(comm_overhead)

        sum_inference_latencys = sum(inference_latencys)
        sum_comm_overheads = sum(comm_overheads)

        self.logger.info(f"In this forward, inference takes {sum_inference_latencys:.1f}s, comm overheads takes: {sum_comm_overheads:.1f}s")






        start = time.time()
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        final_layer_latency = (time.time() - start)
        self.logger.info(f"final layer latency: {final_layer_latency:.3f} s")


        next_cache = next_decoder_cache if use_cache else None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), sum_inference_latencys, sum_comm_overheads