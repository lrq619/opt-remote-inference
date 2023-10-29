from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoder, OPTDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers import OPTForCausalLM
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from typing import Optional, Union, Tuple, List
import pickle
import time
import copy
from .config import FORWARD_PORT, MODEL_NAME
from .logger import init_logger
from .utils import get_object_size


MAX_SEND_SIZE = -1 # disable the size limite
MAX_RECEIVE_SIZE = -1 # disable the size limite

class RemoteOPTDecoderLayers(nn.Module):
    def __init__(self, config, layers_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers_range = layers_range 
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in layers_range]).half()

        whole_model = OPTForCausalLM.from_pretrained(f"facebook/{MODEL_NAME}").half()
         
        layers = whole_model.get_decoder().layers[layers_range[0]:layers_range[-1]+1]

        self.layers.load_state_dict(layers.state_dict())
        self.layers.to("cuda:0")
        self.layers.eval()
        
        del whole_model

        self.logger = init_logger(MODEL_NAME)

    def forward(self, hidden_states, attention_mask, past_key_values):

        self.logger.info(f"received decoder input size:{get_object_size((hidden_states, attention_mask, past_key_values))}")

        hidden_states = hidden_states.to('cuda:0')
        attention_mask = attention_mask.to('cuda:0')
        past_key_values = past_key_values.to('cuda:0') if past_key_values else None

        forward_start = time.time()

        use_cache = True

        next_decoder_cache = () if use_cache else None
        start = time.time() 
        with torch.no_grad():
            for idx, decoder_layer in enumerate(self.layers):
                layer_idx = idx + self.layers_range[0]
                past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

                # decoder_layer_inputs["past_key_value"] = past_key_value
                layer_outputs = decoder_layer(
                    hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    past_key_value = past_key_value,
                    use_cache = use_cache,
                    output_attentions=True,
                    layer_head_mask = None
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[1],)


        inference_latency = (time.time() - start)

        
        self.logger.info(f"inference latency: {inference_latency:.1f} s")

        # send the outputs via grpc, need to send hidden_states and next_decoder_cache back to cpu
        hidden_states = hidden_states.to('cpu')
        next_decoder_cache = next_decoder_cache.to('cpu')

        whole_forward_latency = (time.time() - forward_start)

        return hidden_states, next_decoder_cache, inference_latency, whole_forward_latency 

