import os
import argparse
import torch.distributed.rpc as rpc
import torch.distributed as dist
import time
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

parser = argparse.ArgumentParser(description="Remote Decoder Layers")
parser.add_argument('--rank', type=int, required=True, help='Rank of the worker')
parser.add_argument('--world-size', type=int, required=True, help="World size")

args = parser.parse_args()
rank = args.rank
world_size = args.world_size

time.sleep(420)
print("try to join master")
rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
print("join master finished")
rpc.shutdown()