from torch.distributed import rpc


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
        return rpc.rpc_sync(rref.owner(), _call_method, args = [method, rref]+list(args), kwargs=kwargs)