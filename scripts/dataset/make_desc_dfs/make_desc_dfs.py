import pandas as pd
from pathlib import Path
import sys
from rdkit import RDLogger
from glob import glob

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

FILE_DIR = str(Path(__file__).parent)
PARENT_DIR = str(Path(__file__).parent.parent)

sys.path.insert(0, PARENT_DIR)
from dataset_functions import Dataset_Formatter

make_training_data=False
make_full_data = True
chunksize = 100000


# MAKING TRAINING DATA SET
if make_training_data:
    print('Generating ChEMBL Data set')
    mk = Dataset_Formatter()

    data = mk._load_data(mol_dir='/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/',
                        filename='raw_ChEMBL_data.csv',
                        pymolgen=False,
                        mol_type='smiles',
                        chunksize=chunksize,
                        retain_ids=True,
                        prefix=None)
    print(data)
    
    print('Generating RDKit descriptors')
    
    desc_chunks, full_chunks = mk._calculate_desc_df(descriptor_set='RDKit')

    print('=============================')
    print(desc_chunks)
    print('=============================')
    print(full_chunks)
    print('=============================')
    print('Filtering Results')

    filt_df = mk._filter_df(chembl=True)

    print('Making final chunks...')
    full, desc = mk._make_final_chunks(chunksize=chunksize,
                                       gen_desc_chunks=True,
                                       descriptor_set='RDKit',
                                       save_full_data=False,
                                       save_desc_data=True,
                                       full_save_path=None,
                                       desc_save_path='/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/rdkit/',
                                       filename='ChEMBL_rdkit_desc')
    print('=============================')
    print(full)
    print('=============================')
    print(desc)
    print('=============================')
    print('Making docking files...')
    

    for i, chunks in enumerate(full):
        ch_docking_df = pd.DataFrame()
        ch_docking_df['SMILES'] = chunks['SMILES']
        ch_docking_df['affinity_dock'] = 'N/A'
        ch_docking_df['docking_time'] = 'N/A'
    print('=============================')
    print(ch_docking_df)
    print('=============================')

    ch_docking_df.to_csv(f'/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/dock/ChEMBL_dock.csv.gz',
                          index='ID', compression='gzip')


    print('Generating Mordred descriptors...')
    
    desc_chunks, full_chunks = mk._calculate_desc_df(descriptor_set='Mordred')

    print('=============================')
    print(desc_chunks)
    print('=============================')
    print(full_chunks)
    print('=============================')
    print('Filtering Results')

    filt_df = mk._filter_df(chembl=True)

    print('=============================')
    print(desc_chunks)
    print('=============================')
    print(full_chunks)
    print('=============================')
    print('Making final chunks...')

    full, desc = mk._make_final_chunks(chunksize=chunksize,
                                       gen_desc_chunks=True,
                                       descriptor_set='Mordred',
                                       save_full_data=False,
                                       save_desc_data=True,
                                       full_save_path=None,
                                       desc_save_path='/users/yhb18174/Recreating_DMTA/datasets/ChEMBL/training_data/desc/mordred/',
                                       filename='ChEMBL_mordred_desc')
    print('=============================')
    print(full)
    print('=============================')
    print(desc)
    print('=============================')
    
# MAKING EXPERIMENT DATA SET
if make_full_data:
    mol_dir = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/raw/'
    filename = 'exp_all.inchi'
    chunksize = 100000
    retain_ids = False
    desc_filepath = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/desc/'
    docking_dir = '/users/yhb18174/Recreating_DMTA/datasets/PyMolGen/docking/'

    premade_chunks = glob(f'/users/yhb18174/Recreating_DMTA/datasets/temp/temp_*.csv.gz')


    mk = Dataset_Formatter()

    print('Loading full data from:\nexp_all.inchi')

    data = mk.LoadData(mol_dir=mol_dir,
                         filename=filename,
                         pymolgen=True,
                         prefix='PMG-',
                         mol_type='smiles',
                         chunksize=chunksize,
                         retain_ids=retain_ids,
                         temp_files=premade_chunks)
    
    print(data)
    
    # print('Generating RDKit descriptors')
    
    # desc_chunks, full_chunks = mk._calculate_desc_df(descriptor_set='RDKit')

    # print('Filtering DF')

    # filt_df = mk._filter_df()

    # print('Making Final Chunks')

    # full, desc = mk._make_final_chunks(chunksize=chunksize,
    #                                    gen_desc_chunks=True,
    #                                    descriptor_set='RDKit',
    #                                    save_full_data=False,
    #                                    save_desc_data=True,
    #                                    full_save_path=None,
    #                                    desc_save_path=desc_filepath + 'rdkit/',
    #                                    filename='pymolgen_rdkit_batch')
    

    # for i, chunks in enumerate(full):
    #     py_docking_df = pd.DataFrame()
    #     py_docking_df['SMILES'] = chunks['SMILES']
    #     py_docking_df['affinity_dock'] = 'N/A'
    #     py_docking_df['docking_time'] = 'N/A'
    #     py_docking_df.to_csv(f'{docking_dir}pymolgen_batch{i+1}.csv.gz',
    #                          index='ID', compression='gzip')
    
    # print('Generating Mordred descriptors')

    # desc_chunks, full_chunks = mk._calculate_desc_df(descriptor_set='Mordred')

    # print('Filtering DF')

    # filt_df = mk._filter_df()

    # print('Making Final Chunks')

    # full, desc = mk._make_final_chunks(chunksize=chunksize,
    #                                    gen_desc_chunks=True,
    #                                    descriptor_set='Mordred',
    #                                    save_full_data=False,
    #                                    save_desc_data=True,
    #                                    full_save_path=None,
    #                                    desc_save_path=desc_filepath+ 'mordred/',
    #                                    filename='pymolgen_mordred_batch')


