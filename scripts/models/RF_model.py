#Importing Modules
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import sys
sys.path.insert(0, '/users/yhb18174/scripts')
from smi_funcs import process_smiles

#Directory path 
DIR_PATH= '/users/yhb18174/datasets'

#Dataframes 
DF= 'ChEMBL_docking_dataset.csv'

#Importing Data
DATA= pd.read_csv(f'{DIR_PATH}/{DF}')

#y_column
y_column= ''

#Index
INDEX = 'ID'

#Saving Data (change last part to desired file name)
DESCRIPTOR_FILENAME = f'{DIR_PATH}/results/descriptor_dfs/test_1'

#Functions
def process_SMILES():
    #Function to canonicalise SMILES strings
    DATA['Processed_SMILES']= DATA.apply(lambda row: process_smiles(row['Canon_SMILES'], tauto= 'RDKit', phmodel= "OpenEye", ph= 7.4)[0], axis=1)

def generate_descriptors(save_csv):
    #Processing SMILES for descriptors to be generated from
    process_SMILES()
    
    #Calculating RDKit Descriptors
    desc_ls = [x[0] for x in Chem.Descriptors.descList]

    #Number of descriptors
    n_descs = len(desc_ls)
    
    #Set up descriptor calculator for all RDKit Descriptors
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_ls)

    #Function to generate rdKit descriptors for each ligand
    descs = np.zeros((len(DATA), n_descs))
    for i, smi in enumerate(DATA['Processed_SMILES']):
        
        #Converting SMILE into mol
        mol= Chem.MolFromSmiles(smi)
        
        descs[i]= np.array(calc.CalcDescriptors(mol))
        
        #If 'Ipc' descriptor is calculated, it is refactored to avoid problems with large molecules
        if 'Ipc' in desc_ls:
            descs[i][desc_ls.index('Ipc')]= Chem.GraphDescriptors.Ipc(mol, avg= True)
        
        #Making a descriptor Dataframe
    descriptor_df = pd.DataFrame(descs, columns= desc_ls, index= DATA['ID'])
        
    #Removing descriptors which are common
    del_descs = []
    for desc in descriptor_df.columns:
        #Checking for common values of descriptors
        if np.all(descriptor_df[desc][0] == descriptor_df[desc]) == True:
            del_descs.append(desc)
    descriptor_df= descriptor_df.drop(columns=del_descs)
    
    #Saving the Descriptor Dataframe as csv
    if save_csv==True:
        descriptor_df.to_csv(f'{DESCRIPTOR_FILENAME}.csv')



    
