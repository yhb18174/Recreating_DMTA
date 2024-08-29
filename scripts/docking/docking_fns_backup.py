import pandas as pd
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import re
from pathlib import Path
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool
import os
import random as rand

# Import Openeye Modules
from openeye import oechem
from openeye import oequacpac, oeomega

PROJ_DIR = str(Path(__file__).parent.parent.parent)

# Find Openeye licence
try:
    print("license file found: %s" % os.environ["OE_LICENSE"])
except KeyError:
    print("license file not found, please set $OE_LICENSE")
    sys.exit("Critical Error: license not found")

def WaitForDocking(
    dock_csv: str,
    idxs_in_batch: list,
    check_interval: int=30,
    scores_col: str='CNN_affinity'
):
    while True:
        dock_df = pd.read_csv(dock_csv, index_col='ID', dtype=str)
        pending_docking = dock_df[dock_df[scores_col] == 'PD']

        if pending_docking.empty:
            print('All docking scores present')
            break
        
        print(f'Waiting for the following molecules to dock:\n{list(pending_docking.index)}')
        time.sleep(check_interval)

def GetUndocked(
        dock_df: pd.DataFrame,
        idxs_in_batch: list,
        scores_col: str='CNN_affinity'
):
    df = dock_df.loc[idxs_in_batch]
    undocked = df[df[scores_col].isna()]
    
    return undocked


class Run_GNINA:
    """
    Description
    -----------
    Class to do all things GNINA docking
    """

    def __init__(
        self,
        docking_dir: str,
        molid_ls: list,
        smi_ls: list,
        receptor_path: str,
        center_x: float = 14.66,
        center_y: float = 3.41,
        center_z: float = 10.47,
        size_x: float = 17.67,
        size_y: float = 17.00,
        size_z: float = 13.67,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        cpu: int = 1,
        addH: int = 0,
        stripH: int = 1,
        seed: int = 19,
        cnn_scoring: str = "rescore",
        gnina_path: str = f"{PROJ_DIR}/scripts/docking/gnina",
        env_name: str = "phd_env",
    ):
        """
        Description
        -----------
        Initialisation of the class, setting all of the GNINA values required for the docking.
        Makes directories for each molecule ID.
        By default uses the CNN_Affinity rescored version of the docking scores.

        Parameters
        ----------
        docking_dir (str)           Directory where all of the docking will take place (particularly saving the results to)
        molid_ls (list)             List of molecule ids which will be used to save the data under
        smi_ls (list)               List of SMILES strings which you wish to find the docking score of
        receptor_path (str)         Pathway to the receptor
        center (x,y,z) (float)      X, Y, Z coordinate of the center of the focussed box
        size (x, y, z) (float)      Size of the focussed box in the X, Y, Z dimensions
        exhaustivenedd (int)        Exhaustiveness of the global search
        num_modes (int)             Number of binding modes to generate
        cpu (int)                   Number of CPUs to use (default keep as 1)
        addH (int)                  Automatically adds hydrogens in ligands (0= off, 1= on)
        seed (int)                  Setting the random seed
        cnn_scoring (str)           Determines at what point the CNN scoring function is used:
                                    none = No CNNs used for dockingm uses empirical scoring function
                                    rescore = CNN used for reraning of final poses (least compuationally expensive CNN option)
                                    refinement = CNN to refine poses after Monte Carlo chains (10x slower than rescore)
                                    all = CNN used as scoring dunction throughout (extremely computationally expensive)
        gnina_path (str)            Path to the gnina executable file
        env_name (str)              Name of conda environment to use

        Returns
        -------
        Initialised class
        """

        self.tar_mol_dir_path_ls = [f"{docking_dir}{molid}.tar.gz" for molid in molid_ls]
        self.mol_dir_path_ls = [f"{docking_dir}{molid}/" for molid in molid_ls]

        for dir, tar_dir in zip(self.mol_dir_path_ls, self.mol_dir_path_ls):
            if Path(dir).exists() or Path(tar_dir).exists():
                print(f"{dir} exists")
            else:
                Path(dir).mkdir()

        self.docking_dir = docking_dir
        self.smi_ls = smi_ls
        self.molid_ls = molid_ls
        self.receptor_path = receptor_path
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.num_cpu = cpu
        self.addH = addH
        self.stripH = stripH
        self.seed = seed
        self.cnn_scoring = cnn_scoring
        self.gnina_path = gnina_path
        self.env_name = env_name

    def _mol_enhance_non_opt(self, smi: str, sdf_fpath: str, num_confs: int=10):
        """
        Description
        -----------
        Function to make .sdf file from SMILES string

        Parameters
        ----------
        smi (str)       SMILES string of molecule to enhance
        sdf_fpath (str) File path/name to save the .sdf under

        """
        mol = Chem.MolFromSmiles(smi)
        h_mol = Chem.AddHs(mol)

        with open(sdf_fpath, "w") as file:
            file.write(Chem.MolToMolBlock(h_mol))

    def _mol_enhance(self, smi: str, sdf_fpath: str, num_confs: int=10):

        mol=oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smi)
        oechem.OEAddExplicitHydrogens(mol)

        omega = oeomega.OEOmega()
        omega.SetMaxConfs(num_confs)

        print(f"Number of conformers generated: {mol.NumConfs()}")

        with oechem.oemolostream(sdf_fpath) as ofs:
            oechem.OEWriteMolecules(ofs, mol)


    def _mol_prep(self, smi: str, molid: str, mol_dir: str):
        """
        Description
        -----------
        Prepares the SMILES string under pH 7.4 conditions

        Parameters
        ----------
        smi (str)           SMILES string of molecule to set to pH 7.4
        molid (str)         ID of molecule to save .sdf file as
        mol_dir (str)       Path to directory containing the molecule (set in initialisation)

        Returns
        -------
        Pathway to converted .sdf file
        """

        lig_sdf_path = mol_dir + molid + ".sdf"

        # Convert file paths to Path objects
        ligand_sdf_path = Path(lig_sdf_path)
        before_pH_adjustment_path = ligand_sdf_path.with_name(
            ligand_sdf_path.stem + "_before_pH_adjustment.sdf"
        )
        pH74_path = ligand_sdf_path.with_name(ligand_sdf_path.stem + "_pH74.sdf")

        self._mol_enhance(smi, lig_sdf_path)

        try:
            ifs = oechem.oemolistream()
            ifs.open(str(lig_sdf_path))

            ofs = oechem.oemolostream()
            ofs.SetFormat(oechem.OEFormat_SDF)
            ofs.open(str(pH74_path))

            mol = oechem.OEGraphMol()

            while oechem.OEReadMolecule(ifs, mol):
                if oequacpac.OESetNeutralpHModel(mol):
                    oechem.OEAddExplicitHydrogens(mol)
                    oechem.OEWriteMolecule(ofs, mol)
            ifs.close()
            ofs.close()

            # Rename files using Path methods
            ligand_sdf_path.rename(before_pH_adjustment_path)
            pH74_path.rename(ligand_sdf_path)

        except Exception as e:
            print(f"Failed to convert mol to pH 7.4 for the following reason:\n{e}")

        return pH74_path

    def _make_ph74_sdfs(self):
        """
        Description
        -----------
        Wrapper function to carry out _mol_prep() on self.sdf_path_ls

        Parameters
        ----------
        None

        Returns
        -------
        List of pH 7.4 .sdf file paths
        """

        self.sdf_path_ls = []
        for molid, smi, molid_dir in zip(
            self.molid_ls, self.smi_ls, self.mol_dir_path_ls
        ):
            mol_dir_path = Path(molid_dir)

            if mol_dir_path.suffixes == ['.tar', '.gz']:
                subprocess.run(["tar", "-xzf", molid_dir], check=True)
                path = self._mol_prep(smi=smi, molid=molid, mol_dir=molid_dir)
                self.sdf_path_ls.append(str(path))
            else:
                path = self._mol_prep(smi=smi, molid=molid, mol_dir=molid_dir)
                self.sdf_path_ls.append(str(path))

        return self.sdf_path_ls

    def _make_sdfs(self):
        """
        Description
        -----------
        Simplified SMILES to .sdf conversion without pH considerations

        Parameters
        ----------
        None

        Returns
        -------
        List of .sdf file paths
        """
        self.sdf_path_ls = []

        for molid, smi, molid_dir in zip(
            self.molid_ls, self.smi_ls, self.mol_dir_path_ls
        ):

            # Convert SMILES to RDKit Mol
            mol = Chem.MolFromSmiles(smi)

            if Path(molid_dir + molid + ".sdf").exists():
                print(f"{molid}.sdf file exists")

            if mol is not None:
                # Write to SDF file
                with Chem.SDWriter(molid_dir + molid + ".sdf") as writer:
                    writer.write(mol)
                self.sdf_path_ls.append(molid_dir + molid + ".sdf")
            else:
                print(f"Invalid SMILES string: \n{smi}")

        return self.sdf_path_ls

    def _create_submit_script(
        self,
        molid: str,
        max_time: int,
        mol_dir: str,
        sdf_filename: str,
        output_filename: str,
        log_filename: str,
    ):
        """
        Description
        -----------
        Function to create and submit the gnina submission shell script using the input parameters

        Parameters
        ----------
        molid (str)             ID of the molecule to dock
        max_time (int)          Maximum time which the docking can run for (has to be a whole number between 1 and 168)
        mol_dir (str)           Pathway to the molecule docking directory
        sdf_filename (str)      Filename and path to the .sdf file to dock
        output_filename (str)   Name to save the output .sdf under
        log_filename (str)      Name to save the output .log file under

        Returns
        -------
        Job submission ID for docking job
        """

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

        script_name = str(Path(mol_dir)) + "/" f"{molid}_docking_script.sh"

        with open(script_name, "w") as file:
            file.write(gnina_script)

        subprocess.run(["chmod", "+x", script_name])

        try:
            result = subprocess.run(
                ["sbatch", script_name], capture_output=True, text=True, check=True
            )
            jobid = re.search(r"Submitted batch job (\d+)", result.stdout).group(1)
            return jobid
        except subprocess.CalledProcessError as e:
            print(f"Error in submitting job: {e}")
            return None

    def _submit_jobs(self, max_time: int, username: str = "yhb18174"):
        """
        Description
        -----------
        Function to submit numerous docking jobs in parallel.

        Parameters
        ----------
        max_time (int)      Maximum time to run the docking for
        username (str)      Username which the submitted scripts will be under
        """

        start_time = time.time()

        with Pool() as pool:
            results = pool.starmap(
                self._create_submit_script,
                [
                    (
                        molid,
                        max_time,
                        mol_dir,
                        f"{molid}.sdf",
                        f"{molid}_pose.sdf",
                        f"{molid}.log",
                    )
                    for molid, mol_dir in zip(self.molid_ls, self.mol_dir_path_ls)
                ],
            )

        job_ids = [jobid for jobid in results if jobid is not None]

        timeout = max_time * 60 * 60

        while True:
            dock_jobs = []
            squeue = subprocess.check_output(["squeue", "--users", username], text=True)
            lines = squeue.splitlines()

            job_lines = {
                line.split()[0]: line for line in lines if len(line.split()) > 0
            }

            # Find which submitted job IDs are still in the squeue output
            dock_jobs = set(job_id for job_id in job_ids if job_id in job_lines)

            if not dock_jobs:
                print("All docking done")
                all_docking_scores, top_scores = self._make_docking_csvs(
                    molid_ls=self.molid_ls, save_data=True
                )
                self._compress_files()
                return self.molid_ls, top_scores

            runtime = time.time() - start_time
            if runtime > timeout:
                print("Timeout reached")
                break

            if len(dock_jobs) < 10:
                job_message = f'Waiting for the following jobs to complete: {", ".join(dock_jobs)}'
            else:
                job_message = f"Waiting for {len(dock_jobs)} jobs to finish"

            print(f"\r{job_message.ljust(80)}", end="")
            time.sleep(20)

    def _make_docking_csvs(self, molid_ls: list, save_data: bool = False):
        """
        Description
        -----------
        Function to tak the .log files output from GNINA and make a .csv.gz file containing these docking scores

        Parameters
        ----------
        molid_ls (list)         List of molecule IDs to find the molecule directories
        save_data (bool)        Flag to save the made .csv.gz files

        Returns
        -------
        1: List of docking pd.DataFrames
        2: List of the highest scores for each docked molecule
        """

        df_ls = []
        top_score_ls = []
        for mol_dir, molid in zip(self.mol_dir_path_ls, molid_ls):

            with open(mol_dir + molid + ".log", "r") as file:
                lines = file.readlines()

            for n, line in enumerate(lines):
                if line.startswith("mode |"):
                    start_idx = n
            df_lines = lines[start_idx + 3 :]

            pose_ls = []
            aff_ls = []
            intra_ls = []
            cnn_pose_score_ls = []
            cnn_aff_ls = []

            for l in df_lines:
                items = re.split(r"\s+", l)
                pose_ls.append(items[1])
                aff_ls.append(items[2])
                intra_ls.append(items[3])
                cnn_pose_score_ls.append(items[4])
                cnn_aff_ls.append(items[5])

            docking_df = pd.DataFrame(
                data={
                    "Pose_Number": pose_ls,
                    "Affinity(kcal_mol)": aff_ls,
                    "Intramol(kcal/mol)": intra_ls,
                    "CNN_Pose_Score": cnn_pose_score_ls,
                    "CNN_affinity": cnn_aff_ls,
                }
            ).set_index("Pose_Number")

            docking_df.sort_values(by="CNN_affinity", ascending=False, inplace=True)

            if save_data:
                docking_df.to_csv(
                    f"{mol_dir}{molid}_all_scores.csv.gz",
                    compression="gzip",
                    index="Pose_Number",
                )

            df_ls.append(docking_df)
            top_score_ls.append(docking_df["CNN_affinity"].iloc[0])

        return df_ls, top_score_ls

    def _compress_files(self):
        """
        Description
        -----------
        Function to compress each of the created molecule directory to save data

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        for mol_dir in self.mol_dir_path_ls:
            # Removing the '/'
            mol_dir_path = Path(mol_dir)
            mol_dir_name = str(mol_dir_path.name)

            subprocess.run(
                [
                    "tar",
                    "-czf",
                    f"{mol_dir_name}.tar.gz",
                    "--remove-files",
                    mol_dir_name,
                ],
                cwd=self.docking_dir,
                check=True,
            )