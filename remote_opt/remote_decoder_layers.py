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
from .config import FORWARD_PORT, MODEL_NAME, STATE_DICT_PATH
from .logger import init_logger
from .utils import get_object_size, send_past_key_value_to


MAX_SEND_SIZE = -1 # disable the size limite
MAX_RECEIVE_SIZE = -1 # disable the size limite

class RemoteOPTDecoderLayers(nn.Module):
    def __init__(self, config, layers_range, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.logger = init_logger(MODEL_NAME)
        self.logger.info(f"Enter remote layers loading")
        self.layers_range = layers_range 
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in layers_range]).half()

        # whole_model = OPTForCausalLM.from_pretrained(f"facebook/{MODEL_NAME}").half()
         
        # layers = whole_model.get_decoder().layers[layers_range[0]:layers_range[-1]+1]

        
        save_dict_path = f"{STATE_DICT_PATH}/{MODEL_NAME}/layers-{layers_range[0]}-{layers_range[-1]}.pth"

        self.layers.load_state_dict(torch.load(save_dict_path))
        self.layers.to("cuda:0")
        self.layers.eval()
        
        # del whole_model
        self.logger.info(f"Leaving remote layers loading")


    # only to ensure that this class has been inited in the remote_end
    def is_initialized(self):
        return

    def forward(self, hidden_states, attention_mask, past_key_values):

        self.logger.info(f"received decoder input size:{get_object_size((hidden_states, attention_mask, past_key_values))}")

        hidden_states = hidden_states.to('cuda:0')
        attention_mask = attention_mask.to('cuda:0')
        past_key_values = send_past_key_value_to(past_key_values, 'cuda:0')
        # if past_key_values != None:
        #     for tensor in past_key_values:
        #         print(tensor.shape)
        # if past_key_values != None:
        #     for past_key_value in past_key_values:
        #         for tensor in past_key_value:
        #             tensor = tensor.to('cuda:0')
        # past_key_values = tuple(tensor.to('cuda:0') for tensor in past_key_values) if past_key_values else None
        # past_key_values = past_key_values.to('cuda:0') if past_key_values else None

        forward_start = time.time()

        use_cache = True
        output_attentions = True

        next_decoder_cache = () if use_cache else None
        start = time.time() 
        with torch.no_grad():
            for idx, decoder_layer in enumerate(self.layers):
                layer_idx = idx + self.layers_range[0]
                past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
                layer_outputs = decoder_layer(
                    hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    past_key_value = past_key_value,
                    use_cache = use_cache,
                    output_attentions=output_attentions,
                    layer_head_mask = None
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)


        inference_latency = (time.time() - start)

        
        self.logger.info(f"inference latency: {inference_latency:.1f} s")

        # send the outputs via grpc, need to send hidden_states and next_decoder_cache back to cpu
        hidden_states = hidden_states.to('cpu')
        next_decoder_cache = send_past_key_value_to(next_decoder_cache, 'cpu')


        # for past_key_value in next_decoder_cache:
        #     for tensor in past_key_value:
        #         tensor = tensor.to('cpu')        

        
        # next_decoder_cache = tuple(tensor.to('cpu') for tensor in next_decoder_cache)
        # next_decoder_cache = next_decoder_cache.to('cpu')
        # for tensor in next_decoder_cache:
        #     print(f"after to cpu, type: {type(tensor)}")

        whole_forward_latency = (time.time() - forward_start)

        return hidden_states, next_decoder_cache, inference_latency, whole_forward_latency 

