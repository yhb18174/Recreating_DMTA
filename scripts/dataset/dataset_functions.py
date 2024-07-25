import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from mordred import Calculator, descriptors
import openeye as oe
from openeye import oechem
import openbabel as ob
import numpy as np
from multiprocessing import Pool, cpu_count, Process
from pathlib import Path
import tempfile
from io import StringIO
import subprocess
import glob
import time
import gc
import os

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

ROOT_DIR = Path(__file__).parent
PROJ_DIR = str(Path(__file__).parent.parent.parent)


class Dataset_Formatter():
    def __init__(self):
        
        return
    
    def _process_line(self,
                        line):
        mol = Chem.MolFromInchi(line.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return None
        
    def _make_chunks(self,
                     mol_dir: str,
                     filename: str,
                     retain_ids: bool,
                     pymolgen: bool=False,
                     prefix: str='HW-',
                     chunksize:int=100000):
        print('Making Chunks')

        chunks=[]
        
        if pymolgen:
            with open(f'{mol_dir}/{filename}', 'r') as file:
                lines = file.readlines()
                print('File open')

            lines_chunks = [lines[i:i+chunksize] for i in range(0, len(lines), chunksize)]

            for chunk_idx, chunk_lines in enumerate(lines_chunks):
                chunk_results = []
                id_ls = []

                for line_idx, line in enumerate(chunk_lines):
                    result = self._process_line(line)
                    if result is not None:
                        chunk_results.append(result)
                        id_ls.append(f'{prefix}{len(chunk_results)}')

                chunk_df = pd.DataFrame({'ID': id_ls,
                                         'SMILES': chunk_results})
                chunk_df.set_index('ID', inplace=True)
                chunks.append(chunk_df)                
                print(f'Chunk {chunk_idx + 1} made ({len(chunk_df)} lines)...')

        else:
            mol_df = pd.read_csv(mol_dir + filename)
            smi_ls = mol_df['SMILES'].tolist()
            id_ls = mol_df['ID'].tolist() if retain_ids else [f'{prefix}{i+1}' for i in range(len(smi_ls))]

            for i in range(0, len(smi_ls), chunksize):
                chunk_smi_ls = smi_ls[i:i + chunksize]
                chunk_id_ls = id_ls[i:i + chunksize]
                chunk_df = pd.DataFrame({'ID': chunk_id_ls, 'SMILES': chunk_smi_ls})
                chunk_df.set_index('ID', inplace=True)
                chunks.append(chunk_df)     
        
        print(f'Made {len(chunks)} chunks')
        
        return chunks
    
    def _make_chunks_wrapper(self,
                             args):
        return self._make_chunks(*args)
    
    def _process_mols(self,
                        mol_type: str,
                        filename: str,
                        column: str='SMILES',
                        sub_point: int=None,
                        core: str=None,
                        keep_core: bool=True):
            
            """
            Description
            -----------
            Function that takes a list of inchis and canonicalises them

            Parameters
            ----------
            inchi_list (str)        List of inchi strings you wish to convert to their canonical tautomer
            sub_point (int)         Atom number the fragments were added on to (for keeping original core tautomer)
            core (str)              Core you started with and want to maintain
            keep_core (bool)        Flag to set whether or not you want to maintain the previously defined core 
                                    after canonicalisation

            Returns
            -------
            3 lists:
                    canon_mol_list = List of canonical RDKit mol objects
                    frag_smi_list  = List of the canonical fragments added onto the original core
                                    (only works if keep_core is True, core and sub point are defined)
                    canon_smi_list = List of canonical SMILES strings, determined by RDKit
            """
            
            # Converting all inchis into RDKit mol objects

            mol_converters = {
                'inchi' : Chem.MolFromInchi,
                'smiles' : Chem.MolFromSmiles
            }

            mol_converter = mol_converters.get(mol_type)

            if mol_converter is None:
                raise ValueError("Unsupported mol_type. Supported types are 'inchi' and 'smiles'.")
            
            mol_list = pd.read_csv(filename, compression='gzip')[column]
            
            mol_list = [mol_converter(x) for x in mol_list if mol_converter(x) is not None]

            # Canonicalising the SMILES and obtaining the 3 lists produced by the _canonicalise_smiles() function
            results = [self._canonicalise_smiles(mol, keep_core=keep_core, core_smi=core, sub_point=sub_point) for mol in mol_list]
            
            # Isolating each output list
            frag_smi_list = [result[1] for result in results if results[0]]
            canon_smi_list = [result[2] for result in results if results[0]]
            canon_mol_list = [result[3] for result in results if results[0]]
            kekulised_smi_ls = [self._kekulise_smiles(mol) for mol in canon_mol_list]

            return canon_mol_list, frag_smi_list, canon_smi_list, kekulised_smi_ls

    def _process_mols_wrapper(self,
                              args):
        return self._process_mols(*args)

    def _canonicalise_smiles(self,
                             mol: object,
                             keep_core=True,
                             core_smi: str=None,
                             sub_point: int=None):
        """
        Description
        -----------
        Canonicalising function which deals with tautomeric forms different from that of the original core.
        Here the idea is to canonicalise with RDKit and check the molecule against the original core.
        If original core is maintained, the function returns the canonical SMILES string.
        If not, the fragment is isolated, canonicalised and stiched back onto the original core

        Parameters
        ----------
        smiles (str)            SMILES string of molecule to be checked
        keep_core (bool)        Flag to keep the original specied core, or allow it 
                                to be tautomerised
        core (str)              SMILES string of the core you want to check against
        sub_point (int)         Point to stitch the fragment onto the sub_core
        sub_core (str)          SMILES string of the core you want to add the fragments onto
                                (leave empty is this is just the previous core)

        Returns
        -------        
        has_core (bool)             Tells you whether or not the molecule has one of the tautomeric forms of 
                                    the specified. True means that one of the forms is present
        canon_smi (str)             SMILES string of the canonical form of the molecule after added onto
                                    sub_core
        frag_smi (str)              SMILES string of the fragment added onto the core
        """

        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)
        canon_smi = Chem.MolToSmiles(canon_mol)

        # If maintaining the original core is not an issue then just canonicalise as usual,
        # Does not show fragment SMILES strings
        if not keep_core:
            return True, None, canon_smi, canon_mol
        
        if core_smi is None:
            raise ValueError("Invalid core SMILES string provided.")

        core_mol = Chem.MolFromSmiles(core_smi)        
            
        # Initialising the enumerator, canonicalising the molecule and obtaining all tautomeric
        # forms of the original core
        core_tauts = enumerator.Enumerate(core_mol)


        # Checking if the mol has the original core
        if canon_mol.HasSubstructMatch(core_mol):
            # If so, return the canonical smiles, fragment and core flag
            frag_mol = Chem.ReplaceCore(mol, core_mol)
            frag_smile = Chem.MolToSmiles(frag_mol)

            return True, frag_smile, canon_smi, canon_mol
            
        # If it doesnt have the original core, check the tautomeric forms
        for taut in core_tauts:
            
            # If it has one of the tautometic forms, substitute the core with
            # a dummy atom
            if canon_mol.HasSubstructMatch(taut):
                frag_mol=Chem.ReplaceCore(mol, taut)
                dummy_idx = next(atom.GetIdx() for atom in frag_mol.GetAtoms() is atom.GeySymbol() == '*')
                neighbour_idx = next(bond.GetOtherAtomIdx(dummy_idx) for bond in frag_mol.GetAtomWithIdx(dummy_idx).GetBonds())
                
                frag_mol = Chem.EditableMol(frag_mol)
                frag_mol.RemoveAtom(dummy_idx)
                frag_mol = frag_mol.GetMol()
                frag_smi = Chem.MolToSmiles(frag_mol)

                combined_mol = Chem.CombineMols(frag_mol, core_smi)
                combined_mol = Chem.EditableMole(combined_mol)
                sub_atom = sub_point + frag_mol.GetNumAtoms() - 1

                combined_mol.AddBond*neighbour_idx, sub_atom, Chem.rdchem.BondType.SINGLE
                final_mol = Chem.RemoveHs(combined_mol.GetMol())
                final_smi = Chem.MolToSmiles(final_mol)

                return True, frag_smi, final_smi, final_mol

        return False, None, canon_smi, canon_mol
    
    def _kekulise_smiles(self, mol):
        Chem.Kekulize(mol)
        return Chem.MolToSmiles(mol, kekuleSmiles=True)
    
    def _conv_df_to_str(self,
                        df: pd.DataFrame,
                        **kwargs):
        """
        Description
        -----------
        Converta a Data Frame to a string
        
        Parameters
        ----------
        **kwargs    Arguments for the pd.DataFrame.to_csv() function
         
        Returns
        -------
        String representation of a dataframe
         
         """

        string = StringIO()
        df.to_csv(string, **kwargs)
        return string.getvalue()
    
    def _apply_lilly_rules(self,
                        df=None,
                        smiles=[],
                        smi_input_filename=None,
                        cleanup=True,
                        run_in_temp_dir=True,
                        lilly_rules_script=\
                        str(ROOT_DIR)+'/Lilly-Medchem-Rules/Lilly_Medchem_Rules.rb'):
        """
        Apply Lilly rules to SMILES in a list or a DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing SMILES
        smiles_col: str
            Name of SMILES column

        Returns
        -------
        pd.DataFrame
            DataFrame containing results of applying Lilly's rules to SMILES, including pass/fail and warnings

        Example
        -------
        >>> apply_lilly_rules(smiles=['CCCCCCC(=O)O', 'CCC', 'CCCCC(=O)OCC', 'c1ccccc1CC(=O)C'])
                    SMILES       SMILES_Kekule  Lilly_rules_pass      Lilly_rules_warning  Lilly_rules_SMILES
        0     CCCCCCC(=O)O        CCCCCCC(=O)O              True        D(80) C6:no_rings        CCCCCCC(=O)O
        1              CCC                 CCC             False     TP1 not_enough_atoms                 CCC
        2     CCCCC(=O)OCC        CCCCC(=O)OCC              True  D(75) ester:no_rings:C4        CCCCC(=O)OCC
        3  c1ccccc1CC(=O)C  CC(=O)CC1=CC=CC=C1              True                     None  CC(=O)CC1=CC=CC=C1
        """
        
        lilly_rules_script_path = Path(lilly_rules_script)

        if not lilly_rules_script_path.is_file():
            raise FileNotFoundError(f"Cannot find Lilly rules script (Lilly_Medchem_Rules.rb) at: {lilly_rules_script_path}")
                
        smi_file_txt = self._conv_df_to_str(df[['Kekulised_SMILES', 'ID']],
                                    sep=' ',
                                    header=False,
                                    index=False)
        # Optionally set up temporary directory:
        if run_in_temp_dir:
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = temp_dir.name + '/'
        else:
            run_dir = './'

        # If filename given, save SMILES to this file:
        if smi_input_filename is not None:
            with open(run_dir+smi_input_filename, 'w') as temp:
                temp.write(smi_file_txt)

        # If no filename given just use a temporary file:
        else:
            # Lilly rules script reads the file suffix so needs to be .smi:
            temp = tempfile.NamedTemporaryFile(mode="w+", 
                                            suffix=".smi", 
                                            dir=run_dir)
            temp.write(smi_file_txt)
            # Go to start of file:
            temp.seek(0)

        # Run Lilly rules script
        lilly_results = \
                subprocess.run([f'cd {run_dir}; ruby {lilly_rules_script} {temp.name}'], 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)

        if lilly_results.stderr.decode('utf-8') != '':
            print('WARNING: {}'.format(lilly_results.stderr.decode('utf-8'))) 
        lilly_results = lilly_results.stdout.decode('utf-8')

        # Process results:
        passes = []
        if lilly_results != '':
            for line in lilly_results.strip().split('\n'):

                # Record warning if given:
                if ' : ' in line:
                    smiles_molid, warning = line.split(' : ')
                else:
                    smiles_molid = line.strip()
                    warning = None
                smiles, molid = smiles_molid.split(' ')
                passes.append([molid, warning, smiles])

        # Get reasons for failures:
        failures = []

        #Maybe change 'r' to 'w+'
        for bad_filename in glob.glob(run_dir+'bad*.smi'):
            with open(bad_filename, 'r') as bad_file:
                for line in bad_file.readlines():
                    line = line.split(' ')
                    smiles = line[0]
                    molid = line[1]
                    warning = ' '.join(line[2:]).strip(': \n')
                    failures.append([molid, warning, smiles])

        # Close and remove tempfile:
        # (Do this even if run in a temporary directory to prevent warning when
        # script finishes and tries to remove temporary file at that point)
        if smi_input_filename is None:
            temp.close()

        if run_in_temp_dir:
            temp_dir.cleanup()
        elif cleanup:
            subprocess.run(['rm -f ok{0,1,2,3}.log bad{0,1,2,3}.smi'], shell=True)

        # Convert to DataFrame:
        df_passes = pd.DataFrame(passes, 
                                columns=['ID', 
                                        'Lilly_rules_warning', 
                                        'Lilly_rules_SMILES'])
        #                .set_index('ID', verify_integrity=True)
        df_passes.insert(0, 'Lilly_rules_pass', True)

        df_failures = pd.DataFrame(failures, 
                                columns=['ID', 
                                            'Lilly_rules_warning', 
                                            'Lilly_rules_SMILES'])
        #                .set_index('ID', verify_integrity=True)

        df_failures.insert(0, 'Lilly_rules_pass', False)

        df_all = pd.concat([df_passes, df_failures], axis=0)  

        df_out = pd.merge(df, df_all, on='ID', how='inner')

        print(df[~df['ID'].isin(df_out['ID'])])

        # Check all molecules accounted for:
        if len(df_out) != len(df):
            raise ValueError('Some compounds missing, {} molecules input, but {} compounds output.'.format(len(df), len(df_out)))

        #df['Lilly_rules_pass'].fillna(False, inplace=True)

        return df_out.set_index('ID')
    
    def _adjust_smi_for_ph(self,
                           smi: str,
                           ph: float=7.4,
                           phmodel: str='OpenEye'):
        """
        Description
        -----------
        Function to adjust smiles strings for a defined pH value

        Parameters
        ----------
        smi (str)       SMILES string you want to adjust
        ph (float)      pH you want to adjust SMILES to
        phmodel (str)   pH model used, use either OpenEye or OpenBabel

        Returns
        -------
        SMILES string of adjusted molecule
        """
        # OpenEye pH model needs a pH of 7.4
        if phmodel == 'OpenEye' and ph != 7.4:
            raise ValueError('Cannot use OpenEye pH conversion for pH != 7.4')

        # Use OpenBabel for pH conversion
        # NEED TO PIP INSTALL OBABEL ON phd_env
        if phmodel == 'OpenBabel':
            ob_conv = ob.OBConversion()
            ob_mol = ob.OBMol()
            ob_conv.SetInAndOutFormats("smi", "smi")
            ob_conv.ReadString(ob_mol, smi)
            ob_mol.AddHydrogens(False, # <- only add polar H
                                True,  # <- correct for pH
                                ph)
            ph_smi = ob_conv.WriteString(ob_mol,
                                         True) # <- Trim White Space
            
            # Check that pH adjusted SMILES can be read by RDKit,
            # if not return original SMILES
            if Chem.MolFromSmiles(ph_smi) is None:
                ph_smi=smi

        # Use OpenEye for pH conversion
        elif phmodel == 'OpenEye':
            mol = oechem.OEGraphMol()
            oechem.OESmilesToMol(mol, smi)
            oe.OESetNeutralpHModel(mol)
            ph_smi = oechem.OEMolToSmiles(mol)

        return ph_smi
    
    def _calculate_oe_LogP(self,
                           smi: str):
        """
        Description
        -----------
        Function to calculate the LogP using OpenEye

        Parameters
        ----------
        smi (str)       SMILES string you wish to calculate LogP for

        Returns
        -------
        The oe_logp of input smile
        """

        # Initialise converter
        mol = oe.OEGraphMol()

        # Check is smile gives valid molecule object
        if not oe.OESmilesToMol(mol, smi):
            print('ERROR: {}'.format(smi))
        else:
            # Calculate logP
            try:
                logp= oe.OEGetXLogP(mol, atomxlogps=None)
            except RuntimeError:
                print(smi)
        return logp

    def _get_descriptors(self,
                         mol,
                         missingVal=None,
                         descriptor_set: str='RDKit'):
    
        """
        Description
        -----------
        Function to get the descriptors for a molecule. Can get RDKit or Mordred descriptors

        Parameters
        ----------
        mol (object)            RDKit molecule object you wish to calculate descriptors for
        missingVal (int)        If descriptor value cannot be calculated, enter missingVal instead (keep as None)
        descriptor_set (str)    Option to obtain RDKit or Mordred descriptors

        Returns
        -------
        Dictionary containing all of the descriptors and their values:

        {
            desc_1: val1,
            desc_2: val2,
            ...
        }

        """
        
        res = {}
        if descriptor_set =='RDKit':
            for name,func in Descriptors._descList:
                # some of the descriptor fucntions can throw errors if they fail, catch those here:
                try:
                    val = func(mol)
                except:
                    # print the error message:
                    import traceback
                    traceback.print_exc()
                    # and set the descriptor value to whatever missingVal is
                    val = missingVal
                res[name] = val

        elif descriptor_set == 'Mordred':
            calc = Calculator(descriptors, ignore_3D=True)

            try:
                desc = calc(mol)

                for name, value in desc.items():
                    res[str(name)] = float(value) if value is not None else missingVal

            except Exception as e:
                traceback.print_exc()
                for descriptor in calc.descriptors:
                    res[str(descriptor)] = missingVal

        return res

    def _get_descriptors_wrapper(self,
                                 args):
        
        return self._get_descriptors(*args)

    def LoadData(self,
                   mol_dir:str,
                   filename: str,
                   prefix:str,
                   pymolgen:bool,
                   mol_type: str,
                   column: str='SMILES',
                   chunksize:int=10000,
                   retain_ids:bool=False,
                   core: str=None,
                   sub_point:int=None,
                   keep_core:bool=False,
                   save_chunks: bool=True,
                   save_name: str='PyMolGen',
                   save_path: str=f'{PROJ_DIR}/datasets/'):
        
        temp_files = []
        self.final_file_ls = []

        chunks = self._make_chunks(mol_dir, filename, retain_ids, pymolgen, prefix, chunksize)

        #Save chunks to temp .csv.gz files
        print('Saving chunks:')
        for i, chunk in enumerate(chunks):
            print({i+1})
            tmp_file = f'{save_path}/temp_chunk_{i+1}.csv.gz'
            chunk.to_csv(tmp_file, compression='gzip')
            temp_files.append(tmp_file)

        # Remove chunk from memory
        del chunks
        gc.collect()

        arguments = [(mol_type, file, column, sub_point, core, keep_core) for file in temp_files]
        
        results = []
        for i, args in enumerate(arguments):
            print(f'Processing chunk {i+1}')
            results.append(self._process_mols_wrapper(arguments))

        # with Pool() as pool:
        #     results = pool.map(self._process_mols_wrapper, arguments)

        chunks = [pd.read_csv(file, index_col='ID', compression='gzip') for file in temp_files]

        for i, (chunk, item) in enumerate(zip(chunks, results)):
            print(f'Processing chunk {i+1}')

            canon_mol_ls, frag_smi_ls, canon_smi_ls, kek_smi_ls = item

            canon_smi_ls = [self._adjust_smi_for_ph(smi, phmodel='OpenEye') for smi in canon_smi_ls]

            data = {
                'ID': chunk.index,
                'Mol': canon_mol_ls,
                'Frag_SMILES': frag_smi_ls,
                'SMILES': canon_smi_ls,
                'Kekulised_SMILES': kek_smi_ls
            }
        
            smi_df = pd.DataFrame(data)
            lilly_smi_df = self._apply_lilly_rules(smi_df)

            for file in temp_files:
                file=Path(file)
                if file.exists():
                    file.unlink()

            if save_chunks:
                save_to = f'{save_path}/{save_name}_{i+1}.csv.gz'
                self.final_file_ls.append(save_to)
                lilly_smi_df.drop(columns=['Mol'], inplace=True)
                lilly_smi_df.to_csv(save_to, index='ID', compression='gzip')

        return self.final_file_ls
    
    def CalcDescriptors(self,
                           df_list: list=None,
                           descriptor_set: str='RDKit',
                           save_desc_df:bool=False,
                           save_path:str=None,
                           filename:str=None):
        """
        Description
        -----------
        Function to calculate descriptors for a whole dataset

        Parameters
        ----------
        df (pd.DataFrame)       pd.DataFrame you want to calculate descriptors for, must have
                                column named 'Mol' containing RDKit mol objects
        descriptor_set (str)    Choose the descriptor set you want to generate in the (_get_descriptors() function)
                                either 'RDKit' or 'Mordred'
        
        Returns
        -------
        DataFrame containing all SMILES and their descriptors in the following format:

         ____ _____ _____________ ________ _________ ______ _______ _____
        | ID | Mol | Frag_SMILES | SMILES | Kek SMI |desc1 | desc2 | ... |
        |____|_____|_____________|________|_________|______|_______|_____|
        | id | mol |     frag    |  smi   | kek smi |val 1 |  val2 | ... |
        |____|_____|_____________|________|_________|______|_______|_____|

        """
        
        
        # Setting up temporary df so to not save over self.smi_df
        if df_list is not None:
            tmp_df_ls = [pd.read_csv(file, index_col='ID', compression='gzip') for file in df_list]
        else:
            tmp_df_ls = [pd.read_csv(file, index_col='ID', compression='gzip') for file in self.final_file_ls]
        
        self.batch_desc_ls = []
        self.batch_full_ls = []

        for tmp_df in tmp_df_ls:
            # Getting the descriptors for each mol object and saving the dictionary
            # in a column named descriptors
            rows = tmp_df.to_dict('records')

            with Pool() as pool:
                desc_mp_item = pool.map(self._get_descriptors_wrapper, [([Chem.MolFromSmiles(x) for x in row['SMILES']], None, descriptor_set) for row in rows])

            # Making a new pd.Dataframe with each descriptor as a column, setting the
            # index to match self.smi_df (or tmp_df)
            tmp_df['Descriptors'] = desc_mp_item

            if descriptor_set=='RDKit':
                desc_df = pd.DataFrame(tmp_df['Descriptors'].tolist(), columns=[d[0] for d in Descriptors.descList])

            elif descriptor_set=='Mordred':
                calc=Calculator(descriptors, ignore_3D=True)
                desc_df = pd.DataFrame(tmp_df['Descriptors'].tolist(), columns=[str(d) for d in calc.descriptors])

            desc_df['ID'] = tmp_df.index.tolist()
            desc_df = desc_df.set_index('ID')

            if 'nARing' in desc_df.columns:
                desc_df.rename(columns={'naRing': 'NumAromaticRings'}, inplace=True)
            if 'MW' in desc_df.columns:
                desc_df.rename(columns={'MW': 'MolWt'}, inplace=True)

            self.batch_desc_ls.append(desc_df)

            # Concatenating the two dfs to give the full set of descriptors and SMILES
            batch_df = pd.concat([tmp_df, desc_df], axis=1, join='inner').drop(columns=['Descriptors'])

            batch_df['oe_logp'] = batch_df['SMILES'].apply(self._calculate_oe_LogP)
            batch_df['PFI'] = batch_df['NumAromaticRings'] + batch_df['oe_logp']

            self.batch_full_ls.append(batch_df)

        if save_desc_df:
            for i, dfs in enumerate(self.batch_desc_ls):
                dfs.to_csv(save_path+filename+f'_{i+1}.csv.gz', index_col='ID', compression='gzip')
                print(f'Saved {filename}_{i+1}')

        return self.batch_desc_ls, self.batch_full_ls

    def FilterMols(self,
                   mw_budget: int=600,
                   n_arom_rings_limit: int=3,
                   PFI_limit: int=8,
                   remove_3_membered_rings: bool=True,
                   remove_4_membered_rings: bool=True,
                   max_fused_ring_count: int=1,
                   pass_lilly_rules: bool=True,
                   chembl:bool=False):
        """
        Description
        -----------
        Function to filter undesirable molecules from the dataset
        
        Parameters
        ----------
        mw_budget (int)                 Setting a molecular weight budget for molecules
        n_arom_rings_limit (int)        Setting a limit for the number of aromatic rings for molecule
        PFI_limit (int)                 Setting a PFI limit for molecules (need to implement after
                                        OpenEye License comes)
        remove_*_membered_rings (bool)  Flag to remove 3 or 4 membered cycles
        pass_lilly_rules (bool)         Flag to check if molecules pass the LillyMedChemRules

        Returns
        -------
        A pd.DataFrame of the same format as the _calculate_desc_df output, but with molecules
        filtered off with the specified filters

        """
        
        self.filtered_df_ls =[]
        for df in self.batch_full_ls:
            if not chembl:
            # Obtaining all molecules which pass the defined filters
                all_passing_mols = df[(df['MolWt'] <= mw_budget) &
                                    (df['NumAromaticRings'] <= n_arom_rings_limit) &
                                    (df['PFI'] <= PFI_limit) &
                                    (df['Lilly_rules_pass'] == pass_lilly_rules)]

                filtered_smi = []
                for index, rows in all_passing_mols.iterrows():
                    for mol in rows['Mol'].GetRingInfo().AtomRings():
                        if (remove_3_membered_rings and len(mol) == 3) or (remove_4_membered_rings and len(mol) == 4):
                                filtered_smi.append(rows['SMILES'])

                filtered_results = all_passing_mols[~df['SMILES'].isin(filtered_smi)]            
            columns_to_drop = ['Mol']

            if chembl:
                filtered_results = df
            
            filtered_results.drop(columns=columns_to_drop, inplace=True)
            self.filtered_df_ls.append(filtered_results)

        return self.filtered_df_ls
    
    def MakeFinalChunks(self,
                     chunksize: int,
                     save_full_data: bool=False,
                     gen_desc_chunks:bool=False,
                     save_desc_data: bool=True,
                     descriptor_set: str='RDKit',
                     full_save_path: str=None,
                     desc_save_path: str=None,
                     filename: str=None,):
        """
        Description
        -----------
        Function to make workable pd.DataFrame chunks of data
    
        Parameters
        ----------
        df (pd.DataFrame)       Data Frame you which to split into chunks
        chunksize (int)         Number of rows you want in each chunk of your Data Frame
        save_data (bool)        Flag to save the chunks
        save_path (str)         Path to save the chunks to
        filename (str)          Name to save chunks as, function will number them for you

        Returns
        -------
        Print statements to show which chunk is being saved and where, and a list of the chunks
        """
        
        if len(self.filtered_df_ls) == 1:
            full_df = self.filtered_df_ls[0]
        else:
            full_df = pd.concat(self.filtered_df_ls)

        full_chunks = [full_df.iloc[i: i + chunksize] for i in range(0, full_df.shape[0], chunksize)]
        for i, chunk in enumerate(full_chunks):
            if save_full_data:
                print(f'Saving chunk {i} to:\n{full_save_path}{filename}')
                chunk.to_csv(f'{full_save_path}{filename}_{i+1}.csv.gz', compression='gzip', index='ID')
        
        if gen_desc_chunks:
            if descriptor_set=='RDKit':
                columns, fns = zip(*Descriptors.descList)
            if descriptor_set=='Mordred':
                calc = Calculator(descriptors, ignore_3D=True)
                desc_names = [str(desc) for desc in calc.descriptors]
                columns = []
                for desc in desc_names:
                    if desc == 'naRing':
                        columns.append('NumAromaticRings')
                    elif desc == 'MW':
                        columns.append('MolWt')
                    else:
                        columns.append(desc)
                

            full_desc = full_df.loc[:, columns]
            full_desc_chunks = [full_desc.iloc[i: i + chunksize] for i in range(0, full_df.shape[0], chunksize)]

            for i, chunk in enumerate(full_desc_chunks):
                if save_desc_data:
                    print(f'Saving chunk {i} to:\n{desc_save_path}{filename}')
                    chunk.to_csv(f'{desc_save_path}{filename}_{i+1}.csv.gz', compression='gzip', index='ID')

        return full_chunks, full_desc_chunks