import os

def create_directory(path):
    """
    Create a directory if it doesn't already exist.

    Parameters:
    - path (str): The path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)