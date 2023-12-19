import pickle

def get_object_size(obj):
    serialized_obj = pickle.dumps(obj)
    size_in_bytes = len(serialized_obj)
    return size_in_bytes

def send_past_key_value_to(past_key_values, device:str):
    if past_key_values == None:
        return None
    new_key_values = ()
    for past_key_value in past_key_values:
        new_key_value = ()
        for tensor in past_key_value:
            new_tensor = tensor.to(device)
            new_key_value += (new_tensor,)

        new_key_values += (new_key_value, )

    del past_key_values
    return new_key_values

def calculate_tensor_bytes(tensor):
    """
    Calculates the number of bytes occupied by a tensor.

    Args:
    tensor (np.ndarray): The tensor for which to calculate the byte size.

    Returns:
    int: The number of bytes occupied by the tensor.
    """
    return tensor.size * tensor.itemsize

def calculate_tuple_bytes(tuple):
    sum = 0
    for tensor in tuple:
        sum += tensor.size * tensor.itemsize
    return sum

def calculate_kv_cache_bytes(kv_cache):
    sum = 0
    # print(f"type of kv_cache: {type(kv_cache)}")
    # print(f"type of element in kv_cache:{type(kv_cache[0])}")
    # print(f"type of element in kv_cache element:{type(kv_cache[0][0])}")
    for tuple in kv_cache:
        for tensor in tuple:
            sum += tensor.numel() * tensor.element_size()
    return sum

