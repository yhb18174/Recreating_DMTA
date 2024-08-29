import pandas as pd
import numpy as np
from glob import glob
import re
import fcntl
import time

# def molid2batchno(molid: str,
#                   prefix: str,
#                   dataset_file: str,
#                   chunksize: int=10000):
#     """
#     Description
#     -----------
#     Function to get the batch which the molecule is in from its ID
    
#     Parameters
#     ----------
#     molid (str)         ID of a molecule
#     prefix (str)        Prefix of the molecule ID
#     dataset_file (str)  Common filename of dataset

#     Returns
#     -------
#     Batch number which the molecule with molid is in
#     """

#     mol_no = int(molid.replace(prefix, ''))
#     file_ls = glob(dataset_file)
#     file_ls.sort(key=lambda x: int(re.search(r'\d+', x).group()))

#     prev_check = 0

#     for n in range(1, len(file_ls)+1):
#         n_mols = n*chunksize
#         if prev_check < mol_no >= n_mols:
#             return n
#         else:
#             prev_check = n_mols

def molid2batchno(molid: str,
                  prefix: str,
                  dataset_file: str,
                  chunksize: int = 100000):
    """
    Description
    -----------
    Function to get the batch which the molecule is in from its ID
    
    Parameters
    ----------
    molid (str)         ID of a molecule
    prefix (str)        Prefix of the molecule ID
    dataset_file (str)  Common filename of dataset
    chunksize (int)    Number of molecules per batch

    Returns
    -------
    Batch number which the molecule with molid is in
    """
    
    # Extract the molecule number from its ID
    mol_no = int(molid.replace(prefix, ''))
    
    # List and sort files
    file_ls = glob(dataset_file)
    file_ls.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Determine the batch number
    batch_number = (mol_no - 1) // chunksize + 1
    
    # Check if batch number exceeds number of available files
    if batch_number > len(file_ls):
        raise ValueError(f"Batch number {batch_number} exceeds the number of available dataset files.")
    
    return batch_number

def lock_file(file_path: str):
    
    """
    Description
    -----------
    Function to lock a file to gain exclusive access to the file. Waits if file is locked
    
    Parameters
    ----------
    path (str)      Path to file
    filename (str)  File to lock
    
    Returns
    -------
    Locked file
    """

    while True:
        try:
            with open(file_path, 'r', newline='') as file:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f'Acquired lock on {file}')
                return file
        except BlockingIOError:
             print(f'File {file} is locked. Waiting...')
             time.sleep(30)


def unlock_file(file: object):
    """
    Description
    -----------
    Function to unlock file locked from lock_file function
    
    Parameters
    ----------
    file (object)       File object to unlock

    Returns
    -------
    Unlocked file
    
    """
    return fcntl.flock(file, fcntl.LOCK_UN)