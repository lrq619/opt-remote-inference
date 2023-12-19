from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM
from remote_opt.config import MODEL_NAME, STATE_DICT_PATH
from preprocess import preprocess_alpaca
from model_loading import baseline_model_loading
import time
import torch
from utils import model_size_in_gb

device="cuda:0"

config, tokenizer, model = baseline_model_loading(MODEL_NAME)


model_size = model_size_in_gb(model)
print(f"this model has {model_size:.1f} GB, start loading from pageable memory")
start = time.time()
model.to(device)
pageable_latency = time.time() - start
pageable_throughput = model_size / pageable_latency
print(f"pageable throughput: {pageable_throughput:.1f} GB/s")

model.cpu()
for param in model.parameters():
    param.data = param.data.pin_memory()

for buffer in model.buffers():
    buffer.pin_memory()

start = time.time()
model.to(device)
pinned_latency = time.time() - start
pinned_throughput = model_size / pinned_latency
print(f"pinned throughput: {pinned_throughput:.1f} GB/s")
