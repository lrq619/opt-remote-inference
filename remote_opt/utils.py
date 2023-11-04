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
