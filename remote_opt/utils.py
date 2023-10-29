import pickle

def get_object_size(obj):
    serialized_obj = pickle.dumps(obj)
    size_in_bytes = len(serialized_obj)
    return size_in_bytes