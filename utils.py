import os
import torch
import torch.nn as nn
import pickle

def create_directory(path):
    """
    Create a directory if it doesn't already exist.

    Parameters:
    - path (str): The path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def model_size_in_gb(model):
    """
    Calculate the size of a PyTorch model in gigabytes (GB).

    :param model: PyTorch model
    :return: Size of the model in GB
    """
    total_size = 0
    for param in model.parameters():
        # Get the number of elements in the parameter
        param_size = param.numel()
        # Get the size of each element in bytes (float32 is 4 bytes)
        element_size = param.element_size()
        # Calculate total size in bytes and add to total
        total_size += param_size * element_size

    # Convert bytes to gigabytes
    size_in_gb = total_size / (1024 ** 3)
    return size_in_gb

def get_output_token_from_generate(generate_ids, tokenizer):
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def calculate_tensor_bytes(tensor):
    """
    Calculates the number of bytes occupied by a tensor.

    Args:
    tensor (np.ndarray): The tensor for which to calculate the byte size.

    Returns:
    int: The number of bytes occupied by the tensor.
    """
    return tensor.numel() * tensor.element_size()

def calculate_object_bytes(obj):
    """
    Calculates the size of a Python object in bytes using serialization with pickle.

    Args:
    obj: The Python object for which to calculate the size.

    Returns:
    int: The size of the serialized object in bytes.
    """
    serialized_obj = pickle.dumps(obj)
    return len(serialized_obj)

def calculate_tuple_bytes(tuple):
    sum = 0
    for tensor in tuple:
        sum += tensor.numel() * tensor.element_size()
    return sum

def calculate_kv_cache_bytes(kv_cache):
    sum = 0
    print(f"type of kv_cache: {type(kv_cache)}")
    print(f"type of element in kv_cache:{type(kv_cache[0])}")
    print(f"type of element in kv_cache element:{type(kv_cache[0][0])}")
    for tuple in kv_cache:
        for tensor in tuple:
            sum += tensor.numel() * tensor.element_size()
    return sum