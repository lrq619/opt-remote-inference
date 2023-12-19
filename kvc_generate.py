from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, LogitsProcessorList
import torch

model_name = "opt-6.7b"
config = AutoConfig.from_pretrained(f"facebook/{model_name}")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}",padding_side='left')
model = OPTForCausalLM.from_pretrained(f"facebook/{model_name}").half()

device = "cuda:0"
model.to(device)
model.eval()

prompt = "This is just a test prompt"
input = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

input_ids = input["input_ids"]

logits_processor = LogitsProcessorList()
past_key_values = None


max_new_tokens = 50 # given the max new tokens that's going to be generated
# generation procedure
for i in range(max_new_tokens):
    model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
    outputs = model.forward(**model_inputs)

    next_token_logits = outputs.logits[:,-1,:]
    past_key_values = outputs.past_key_values # This is the kv-cache
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)


generated_output = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(f"generated output: {generated_output}")