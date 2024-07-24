import pandas as pd
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from datetime import datetime
from pathlib import Path
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROJ_DIR= str(Path(__file__).parent.parent.parent)

class GNINA_fns():
    def __init__(self,
                 receptor_path: str,
                 mol_dir: str,
                 sdf_filename: str,
                 output_filename: str,
                 log_filename: str,
                 center_x: float=14.66,
                 center_y: float=3.41,
                 center_z: float=10.47,
                 size_x: float=17.67,
                 size_y: float=17.00,
                 size_z: float=13.67,
                 exhaustiveness: int=8,
                 num_modes: int=9,
                 cpu: int=1,
                 addH: int=0,
                 stripH: int=1,
                 seed: int=19,
                 cnn_scoring: str='rescore', 
                 gnina_path: str=f'{PROJ_DIR}/scripts/docking/gnina'):
        
        self.receptor_path=receptor_path
        self.mol_dir=mol_dir
        self.sdf_filename=sdf_filename
        self.output_filename=output_filename
        self.log_filename=log_filename
        self.center_x=center_x
        self.center_y=center_y
        self.center_z=center_z
        self.size_x=size_x
        self.size_y=size_y
        self.size_z=size_z
        self.exhaustiveness=exhaustiveness
        self.num_modes=num_modes
        self.num_cpu=cpu
        self.addH=addH
        self.stripH=stripH
        self.seed=seed
        self.cnn_scoring=cnn_scoring
        self.gnina_path=gnina_path

        if Path(self.mol_dir).exists() and Path(self.mol_dir).is_dir():
            print(f'{self.mol_dir} exists')
        else:
            print('Making molecule directory')
            Path(self.mol_dir).mkdir()

        if Path(self.mol_dir+self.log_filename).exists():
            print('Log file already exists')

    def _smiles_to_sdf(self,
                      smi: str,
                      dir:str,
                      filename: str):
        
        #Convert SMILES to RDKit Mol
        mol = Chem.MolFromSMiles(smi)
        if mol is not None:
            # Write to SDF file
            with Chem.SDWriter(dir + '/' + filename) as writer:
                writer.write(mol)
        else:
            print('Invalid SMILES string')
    
    def Run_GNINA(self,
                  max_time: int=10,
                  n_cpus: int=1,
                  env_name: str='phd_env',
                  scriptname: str=None,
                  mol_id: str=None):
        
        gnina_script= f"""\
#!/bin/bash
#SBATCH --export=ALL
#SBATCH --time {max_time}:00:00
#SBATCH --job-name={mol_id}_Docking
#SBATCH --ntasks={n_cpus}
#SBATCH --partition=standard
#SBATCH --account=palmer-addnm
#SBATCH --output={self.mol_dir}slurm-%j.out

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_prologue.sh ]
then
    /opt/software/scripts/job_prologue.sh
fi
#----------------------------------------------------------

module purge
module load anaconda/python-3.9.7

source activate {env_name}
        
{self.gnina_path} \\
    --receptor "{self.receptor_path}" \\
    --ligand "{self.mol_dir}{self.sdf_filename}" \\
    --out "{self.output_filename}" \\
    --log "{self.log_filename}" \\
    --center_x {self.center_x} \\
    --center_y {self.center_y} \\
    --center_z {self.center_z} \\
    --size_x {self.size_x} \\
    --size_y {self.size_y} \\
    --size_z {self.size_z} \\
    --exhaustiveness {self.exhaustiveness} \\
    --num_modes {self.num_modes} \\
    --cpu {self.num_cpu} \\
    --no_gpu \\
    --addH {self.addH} \\
    --stripH {self.stripH} \\
    --seed {self.seed} \\
    --cnn_scoring "{self.cnn_scoring}"

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_epilogue.sh ]
then
    /opt/software/scripts/job_epilogue.sh
fi
#----------------------------------------------------------
"""
        script_file=self.mol_dir+scriptname+'.sh'
        with open(script_file, 'w') as file:
            file.write(gnina_script)
        subprocess.run(['chmod', '+x', script_file])
        subprocess.run(['sbatch', script_file], capture_output=True, text=True)

        finished=False
        timeout=7200
        start_time=time.time()
        wait_time=30
        
        
        # while not finished:
            # try:
            #     df = self._make_docking_csv(mol_dir=self.mol_dir,
            #                            log_file=self.log_filename,
            #                            save_data=False)
            #     docking_score = float(df['CNN_Affinity'].iloc[0])
            #     finished=True
            # except FileNotFoundError:
            #     current_time=time.time()
            #     elapsed_time= current_time - start_time

            #     if elapsed_time > timeout:
            #         print(f'Timeout limit reached:\nFile {self.log_filename} not found.')
            #         break

            #     print(f'File not found:\n{self.log_filename}. Retrying in {wait_time} seconds.')
            #     time.sleep(wait_time)
            
            # except Exception as e:
            #     print(f'An error occurred:\n{e}')
            #     break

        # return docking_score

    def _make_docking_csv(self,
                          mol_dir: str,
                          log_file: str,
                          save_data: bool=False,
                          filename: str=None):
        
        with open(log_file, 'r') as file:
            lines = file.readlines()
        
        for n, line in enumerate(lines):
            if line.startswith('mode |'):
                start_idx=n
        df_lines=lines[start_idx+3:]

        pose_ls =[]
        aff_ls=[]
        intra_ls=[]
        cnn_pose_score_ls=[]
        cnn_aff_ls=[]

        for l in df_lines:
            items = re.split(r'\s+', l)
            pose_ls.append(items[1])
            aff_ls.append(items[2])
            intra_ls.append(items[3])
            cnn_pose_score_ls.append(items[4])
            cnn_aff_ls.append(items[5])

        docking_df = pd.DataFrame(data={
                        'Pose_Number': pose_ls,
                        'Affinity(kcal_mol)': aff_ls,
                        'Intramol(kcal/mol)':intra_ls,
                        'CNN_Pose_Score': cnn_pose_score_ls,
                        'CNN_Affinity': cnn_aff_ls
                        }
                        ).set_index('Pose_Number')
        
        docking_df.sort_values(by='CNN_Affinity', ascending=False, inplace=True)
        
        if save_data:
            docking_df.to_csv(f'{mol_dir}{filename}_all_scores.csv.gz', compression='gzip', index='Pose_Number')

        return docking_df
    
class Run_MP_GNINA():
    def __init__(self,
                 docking_dir: str,
                 molid_ls: list,
                 smi_ls: list,
                 receptor_path: str,
                 center_x: float=14.66,
                 center_y: float=3.41,
                 center_z: float=10.47,
                 size_x: float=17.67,
                 size_y: float=17.00,
                 size_z: float=13.67,
                 exhaustiveness: int=8,
                 num_modes: int=9,
                 cpu: int=1,
                 addH: int=0,
                 stripH: int=1,
                 seed: int=19,
                 cnn_scoring: str='rescore', 
                 gnina_path: str=f'{PROJ_DIR}/scripts/docking/gnina',
                 env_name: str='phd_env'):
        
        
        self.mol_dir_path_ls = [f'{docking_dir}{molid}/' for molid in molid_ls]
        
        for dirs in self.mol_dir_path_ls:
            if Path(dirs).exists() and Path(dirs).is_dir():
                print(f'{dirs} exists')
            else:
                Path(dirs).mkdir()

        self.smi_ls = smi_ls
        self.molid_ls = molid_ls
        self.receptor_path=receptor_path
        self.center_x=center_x
        self.center_y=center_y
        self.center_z=center_z
        self.size_x=size_x
        self.size_y=size_y
        self.size_z=size_z
        self.exhaustiveness=exhaustiveness
        self.num_modes=num_modes
        self.num_cpu=cpu
        self.addH=addH
        self.stripH=stripH
        self.seed=seed
        self.cnn_scoring=cnn_scoring
        self.gnina_path=gnina_path
        self.env_name=env_name


    def _make_sdfs(self):

        sdf_path_ls = []

        for molid, smi, molid_dir in zip(self.molid_ls, self.smi_ls, self.mol_dir_path_ls):
        
            #Convert SMILES to RDKit Mol
            mol = Chem.MolFromSmiles(smi)

            if Path(molid_dir + molid + '.sdf').exists():
                print(f'{molid}.sdf file exists')

            if mol is not None:
            # Write to SDF file
                with Chem.SDWriter(molid_dir + molid + '.sdf') as writer:
                    writer.write(mol)
                sdf_path_ls.append(molid_dir + molid + '.sdf')
            else:
                print(f'Invalid SMILES string: \n{smi}')

        return sdf_path_ls
    
    def _create_submit_script(self,
                             molid: str,
                             max_time: int,
                             mol_dir: str,
                             sdf_filename: str,
                             output_filename: str,
                             log_filename: str):

        gnina_script = f"""\
#!/bin/bash
#SBATCH --export=ALL
#SBATCH --time {max_time}:00:00
#SBATCH --job-name=dock_{molid}
#SBATCH --ntasks={self.num_cpu}
#SBATCH --partition=standard
#SBATCH --account=palmer-addnm
#SBATCH --output={mol_dir}slurm-%j.out

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_prologue.sh ]
then
    /opt/software/scripts/job_prologue.sh
fi
#----------------------------------------------------------

module purge
module load anaconda/python-3.9.7

source activate {self.env_name}
        
{self.gnina_path} \\
    --receptor "{self.receptor_path}" \\
    --ligand "{mol_dir}{sdf_filename}" \\
    --out "{mol_dir}{output_filename}" \\
    --log "{mol_dir}{log_filename}" \\
    --center_x {self.center_x} \\
    --center_y {self.center_y} \\
    --center_z {self.center_z} \\
    --size_x {self.size_x} \\
    --size_y {self.size_y} \\
    --size_z {self.size_z} \\
    --exhaustiveness {self.exhaustiveness} \\
    --num_modes {self.num_modes} \\
    --cpu {self.num_cpu} \\
    --no_gpu \\
    --addH {self.addH} \\
    --stripH {self.stripH} \\
    --seed {self.seed} \\
    --cnn_scoring "{self.cnn_scoring}"

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_epilogue.sh ]
then
    /opt/software/scripts/job_epilogue.sh
fi
#----------------------------------------------------------
"""
        
        script_name = str(Path(mol_dir)) + '/' f"{molid}_docking_script.sh"

        with open(script_name, 'w') as file:
            file.write(gnina_script)

        subprocess.run(['chmod', '+x', script_name])

        try:
            result = subprocess.run(['sbatch', script_name], capture_output=True, text=True, check=True)
            jobid = re.search(r'Submitted batch job (\d+)', result.stdout).group(1)
            return jobid
        except subprocess.CalledProcessError as e:
            print(f'Error in submitting job: {e}')
            return None

    def _submit_jobs(self,
                     max_time,
                     username: str='yhb18174'):

        start_time = time.time()

        with Pool() as pool:
            results = pool.starmap(self._create_submit_script,
                            [(molid, max_time, mol_dir, 
                            f'{molid}.sdf', f'{molid}_pose.sdf', 
                            f'{molid}.log') for molid, mol_dir in zip(self.molid_ls, self.mol_dir_path_ls)])
            
        job_ids = [jobid for jobid in results if jobid is not None]

        timeout=max_time*60*60

        while True:
            dock_jobs = []
            squeue = subprocess.check_output(['squeue', '--users', 'yhb18174'], text=True)
            lines = squeue.splitlines()

            job_lines = {line.split()[0]: line for line in lines if len(line.split()) > 0}

            # Find which submitted job IDs are still in the squeue output
            dock_jobs = set(job_id for job_id in job_ids if job_id in job_lines)

            if not dock_jobs:
                print('All docking done')
                all_docking_scores, top_scores = self._make_docking_csvs(molid_ls=self.molid_ls, save_data=True)
                self._compress_files()
                return self.molid_ls, top_scores

            runtime=time.time()-start_time
            if runtime> timeout:
                print('Timeout reached')
                break
            
            if len(dock_jobs) < 10:
                job_message = f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
            else:
                job_message = f'Waiting for the {len(dock_jobs)} to finish'

            print(f'\r{job_message.ljust(80)}', end='')
            time.sleep(180)

    def _make_docking_csvs(self,
                          molid_ls: list,
                          save_data: bool=False):
        
        df_ls = []
        top_score_ls = []
        for mol_dir, molid in zip(self.mol_dir_path_ls, molid_ls):

            with open(mol_dir+molid+'.log', 'r') as file:
                lines = file.readlines()
        
            for n, line in enumerate(lines):
                if line.startswith('mode |'):
                    start_idx=n
            df_lines=lines[start_idx+3:]

            pose_ls =[]
            aff_ls=[]
            intra_ls=[]
            cnn_pose_score_ls=[]
            cnn_aff_ls=[]

            for l in df_lines:
                items = re.split(r'\s+', l)
                pose_ls.append(items[1])
                aff_ls.append(items[2])
                intra_ls.append(items[3])
                cnn_pose_score_ls.append(items[4])
                cnn_aff_ls.append(items[5])

            docking_df = pd.DataFrame(data={
                            'Pose_Number': pose_ls,
                            'Affinity(kcal_mol)': aff_ls,
                            'Intramol(kcal/mol)':intra_ls,
                            'CNN_Pose_Score': cnn_pose_score_ls,
                            'CNN_Affinity': cnn_aff_ls
                            }
                            ).set_index('Pose_Number')
            
            docking_df.sort_values(by='CNN_Affinity', ascending=False, inplace=True)
            
            if save_data:
                docking_df.to_csv(f'{mol_dir}{molid}_all_scores.csv.gz', compression='gzip', index='Pose_Number')

            df_ls.append(docking_df)
            top_score_ls.append(df_ls['CNN_Affinity'].iloc[0])

        return df_ls, top_score_ls
    
    def _compress_files(self):
        
        for mol_dir in self.mol_dir_path_ls:
            archive = mol_dir[:-1]
            subprocess.run(['tar', '-czf', f'{archive}.tar.gz', '--remove-files', mol_dir], check=True)

    