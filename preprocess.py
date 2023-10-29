from datasets import load_dataset

def preprocess_alpaca(prompt_num):
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = dataset["train"]["instruction"][:prompt_num]
    input = dataset["train"]["input"][:prompt_num]
    output = dataset["train"]["output"][:prompt_num]

    prompts = []
    for i in range(len(instructions)):
        prompt = ""
        if input[i] == "":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instructions[i]}"
        else:
            prompt =  f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {instructions[i]} ### Input:{input[i]}"
        prompts.append(prompt)
    return prompts, output
