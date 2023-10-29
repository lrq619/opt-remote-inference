import os
import torch.distributed.rpc as rpc
import torch.distributed as dist
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
# dist.init_process_group(backend='gloo', rank=1, world_size=2)
rpc.init_rpc("worker1", rank=1, world_size=2)
rpc.shutdown()