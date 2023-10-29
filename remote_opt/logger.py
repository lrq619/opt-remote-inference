from loguru import logger   
import os
import sys

def init_logger(model_name : str):
# init logger
    log_dir = f'res/{model_name}'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'invoker.log')
    with open(log_file, 'w') as f:
        pass
    logger.add(log_file, format="{time:YYYY-MM-DDTHH:mm:ss.SSS} {level} {file}:{line} {message}")
    logger.add(sys.stdout, format="{time:YYYY-MM-DDTHH:mm:ss.SSS} {level} {file}:{line} {message}")
    return logger
