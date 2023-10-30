import subprocess
from config import WORKER_LAYERS_MAP

def main():
    world_size = 1 + len(WORKER_LAYERS_MAP.keys()) 
    master_process = subprocess.Popen(['python', 'master.py', '--world-size', f'{world_size}'])

    for rank in WORKER_LAYERS_MAP.keys():
        worker_process = subprocess.Popen(['python', 'worker.py','--rank',f'{rank}', '--world-size', f'{world_size}'])

    master_process.communicate()

    worker_process.communicate()

if __name__ == '__main__':
    main()