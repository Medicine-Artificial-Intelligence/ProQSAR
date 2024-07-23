from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from map4 import MAP4Calculator
from mhfp.encoder import MHFPEncoder
from cats2d.rd_cats2d import CATS2D
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem import PandasTools, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
from AtQSAR.AtlFeaturizer.Pubchem import calcPubChemFingerAll
from AtQSAR.AtlFeaturizer.featurizer import MolecularFingerprint
from mordred import Calculator, descriptors
from rdkit import Chem


class FeatureGenerator:
    """
    Class for generating molecular features.

    Parameters
    ----------
    data : pandas.DataFrame
        Data with ID, smiles, and activity columns.
    mol_col: str
        Name of the column with molecular structures (e.g., smiles, SMILES, Canonical smiles).
    activity_col: str
        Name of the activity column (e.g., pIC50, pChEMBL Value).
    ID: str
        Name of the identity column.
    save_dir: str
        Directory to save data after calculations.
    n_jobs: int, optional
        The number of CPUs to use for parallel processing. -1 means using all processors.
    verbose: int, optional
        The verbosity level of the joblib parallelization.

    Attributes
    ----------
    data: pandas.DataFrame
        The input data containing molecular information.
    """

    def __init__(self, data, mol_col, activity_col, ID, save_dir, n_jobs=-1, verbose=1):
        self.mol_col = mol_col
        self.activity_col = activity_col
        self.ID = ID
        self.save_dir = save_dir
        self.data = data[[self.ID, self.mol_col, self.activity_col]]
        self.n_jobs = n_jobs
        self.verbose = verbose

    def calculate_rdk_fingerprints(self, rdk="all"):
        """
        Calculate RDKit fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.
        rdk : str, optional
            The type of RDKit fingerprint to calculate. Default is 'all', which calculates fingerprints for RDKit versions 5, 6,
            and 7.

        Returns:
        None
        """
        rdk_mappings = {
            "rdk5": [[5, 2048]],
            "rdk6": [[6, 2048]],
            "rdk7": [[7, 4096]],
            "all": [[5, 2048], [6, 2048], [7, 4096]],
        }
        rdk_type = rdk_mappings.get(rdk, [])
        if not rdk_type:
            raise ValueError("Invalid RDK type specified")

        for maxPath, fpSize in rdk_type:
            print(f"CALCULATING RDK{maxPath} FINGERPRINTS...")

            # Parallel processing
            fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(MolecularFingerprint.RDKFp)(mol, maxPath=maxPath, fpSize=fpSize)
                for mol in self.data[self.mol_col]
            )

            # Convert to DataFrame
            df = pd.DataFrame(fingerprints)

            # Concatenate with original DataFrame
            result = pd.concat([self.data, df], axis=1)

            # Drop unnecessary columns
            result = result.drop([self.mol_col], axis=1)

            # Save to CSV
            result.to_csv(f"{self.save_dir}RDK{maxPath}.csv", index=False)

    def calculate_circular_fingerprints(self, ecfp_fcfp="all"):
        """
        Calculate ECFP (Extended Connectivity Fingerprint) and FCFP (Functional-Class Fingerprints) for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.
        ecfp_fcfp : str, optional
            The type of fingerprint to calculate. Default is 'all', which calculates all available ECFP and FCFP fingerprints.

        Returns:
        None
        """
        # Define the radius and nBits combinations for ECFP and FCFP
        fingerprint_types = {
            "ecfp": [[1, 2048], [2, 2048], [3, 4096]],
            "fcfp": [[1, 2048], [2, 2048], [3, 4096]],
        }

        selected_fingerprints = []
        if ecfp_fcfp == "all":
            selected_fingerprints.extend(
                ["ecfp2", "ecfp4", "ecfp6", "fcfp2", "fcfp4", "fcfp6"]
            )
        elif ecfp_fcfp in ["ecfp2", "ecfp4", "ecfp6", "fcfp2", "fcfp4", "fcfp6"]:
            selected_fingerprints.append(ecfp_fcfp)

        for fingerprint_name in selected_fingerprints:
            if fingerprint_name.startswith("ecfp") or fingerprint_name.startswith(
                "fcfp"
            ):
                fingerprint_type = fingerprint_name[
                    :4
                ]  # Extract fingerprint type (ecfp or fcfp)
                index = int(fingerprint_name[4]) // 2 - 1  # Correct index adjustment
                radius, nBits = fingerprint_types[fingerprint_type][index]
                d = 2 * radius

                # Print radius, nBits, and fingerprint name for debugging
                print(f"Fingerprint Name: {fingerprint_name}")
                print(f"Radius: {radius}")
                print(f"nBits: {nBits}")

                print(f"CALCULATING {fingerprint_name.upper()} FINGERPRINTS...")

                # Parallel processing for ECFP and FCFP fingerprints with useFeatures parameter
                fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    (
                        delayed(MolecularFingerprint.ECFPs)(
                            mol, radius=radius, nBits=nBits, useFeatures=False
                        )
                        if fingerprint_type == "ecfp"
                        else delayed(MolecularFingerprint.ECFPs)(
                            mol, radius=radius, nBits=nBits, useFeatures=True
                        )
                    )
                    for mol in self.data[self.mol_col]
                )

                # Convert to DataFrame
                df = pd.DataFrame(fingerprints)

                # Create a file name based on the fingerprint type and parameters
                fingerprint_file_name = f"{fingerprint_type.upper()}{d}.csv"

                # Concatenate with the original DataFrame
                result = pd.concat([self.data, df], axis=1)

                # Drop unnecessary columns
                result = result.drop([self.mol_col], axis=1)

                # Save to CSV with the correct file name
                result.to_csv(f"{self.save_dir}{fingerprint_file_name}", index=False)

    def calculate_maccs_fingerprints(self):
        """
        Calculate MACCS (Molecular ACCess System) fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        print("CALCULATING MACCs FINGERPRINTS...")
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(MolecularFingerprint.MACCs)(mol) for mol in self.data[self.mol_col]
        )

        # Convert to DataFrame
        df = pd.DataFrame(fingerprints)

        # Concatenate with original DataFrame
        result = pd.concat([self.data, df], axis=1)

        # Drop unnecessary columns
        result = result.drop([self.mol_col], axis=1)

        # Save to CSV
        result.to_csv(f"{self.save_dir}MACCs.csv", index=False)

    def calculate_avalon_fingerprints(self):
        """
        Calculate AVALON (Algorithm for Visualizing And Linking ONtologies) fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        print("CALCULATING AVALON FINGERPRINTS...")
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(MolecularFingerprint.Avalon)(mol) for mol in self.data[self.mol_col]
        )

        # Convert to DataFrame
        df = pd.DataFrame(fingerprints)

        # Concatenate with original DataFrame
        result = pd.concat([self.data, df], axis=1)

        # Drop unnecessary columns
        result = result.drop([self.mol_col], axis=1)

        # Save to CSV
        result.to_csv(f"{self.save_dir}Avalon.csv", index=False)

    def calculate_pubchem_fingerprints(self):
        """
        Calculate PubChem fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        self.pubchem = self.data.copy()
        print("CALCULATING PUBCHEM FINGERPRINTS...")
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(calcPubChemFingerAll)(mol) for mol in self.pubchem[self.mol_col]
        )
        X = np.stack(fingerprints)
        df = pd.DataFrame(X)
        self.pubchem = pd.concat([self.pubchem, df], axis=1).drop(
            [self.mol_col, self.ID], axis=1
        )
        self.pubchem.to_csv(f"{self.save_dir}Pubchem.csv", index=False)

    def calculate_map4_fingerprints(self):
        """
        Calculate MAP4 fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        self.map4 = self.data.copy()
        print("CALCULATING MAP4 FINGERPRINTS...")
        m4_calc = MAP4Calculator(is_folded=True)
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(m4_calc.calculate)(mol) for mol in self.map4[self.mol_col]
        )
        X = np.stack(fingerprints)
        df = pd.DataFrame(X)
        self.map4 = pd.concat([self.map4, df], axis=1).drop([self.mol_col], axis=1)
        self.map4.to_csv(f"{self.save_dir}Map4.csv", index=False)

    def calculate_secfp_fingerprints(self):
        """
        Calculate SECFP fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        self.secfp = self.data.copy()
        print("CALCULATING SECFP FINGERPRINTS...")
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(MHFPEncoder.secfp_from_mol)(mol) for mol in self.secfp[self.mol_col]
        )
        X = np.stack(fingerprints)
        df = pd.DataFrame(X)
        self.secfp = pd.concat([self.secfp, df], axis=1).drop([self.mol_col], axis=1)
        self.secfp.to_csv(f"{self.save_dir}Secfp.csv", index=False)

    def calculate_pharm2dgb_fingerprints(self):
        """
        Calculate 2D pharmacophore (Gobbi) fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        self.gobbi = self.data.copy()
        print("CALCULATING PHARMACOPHORE GOBBI FINGERPRINTS...")
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(MolecularFingerprint.mol2pharm2dgbfp)(mol)
            for mol in self.gobbi[self.mol_col]
        )
        X = np.stack(fingerprints)
        df = pd.DataFrame(X)
        self.gobbi = pd.concat([self.gobbi, df], axis=1).drop([self.mol_col], axis=1)
        self.gobbi.to_csv(f"{self.save_dir}Ph4.csv", index=False)

    def calculate_cats2d_fingerprints(self):
        """
        Calculate CATS2D pharmacophore fingerprints for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        self.cats2d = self.data.copy()
        print("CALCULATING PHARMACOPHORE CATS2D FINGERPRINTS...")
        fingerprints = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(MolecularFingerprint.mol2cats)(mol)
            for mol in self.cats2d[self.mol_col]
        )
        X = np.stack(fingerprints)
        df = pd.DataFrame(X)
        self.cats2d = pd.concat([self.cats2d, df], axis=1).drop([self.mol_col], axis=1)
        self.cats2d.to_csv(f"{self.save_dir}Cats2d.csv", index=False)

    def calculate_rdk_descriptors(self):
        """
        Calculate RDKit molecular descriptors for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """
        self.rdkdes = self.data.copy()
        print("CALCULATING RDKit descriptors...")
        descriptors = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(MolecularFingerprint.RDKDes)(mol)
            for mol in self.rdkdes[self.mol_col]
        )
        X = np.stack(descriptors)
        descriptor_names = [x[0] for x in Descriptors._descList]
        df = pd.DataFrame(X, columns=descriptor_names)
        self.rdkdes = pd.concat([self.rdkdes, df], axis=1).drop([self.mol_col], axis=1)
        self.rdkdes.to_csv(f"{self.save_dir}RDKdes.csv", index=False)

    def calculate_mordred_descriptors(self, ignore_3D=True):
        """
        Calculate Mordred molecular descriptors for molecules.

        Parameters:
        self : Your class instance
            Your class instance that contains molecule data.

        Returns:
        None
        """

        self.mord = self.data.copy()
        Chem.PandasTools.WriteSDF(
            self.mord, self.save_dir + "conf.sdf", molColName=self.mol_col
        )

        print("CALCULATING Mordred descriptors...")

        #!python -m mordred {self.save_dir+'conf.sdf'} -3 -o {self.save_dir+'Mordred.csv'}

        import subprocess

        # Define the command to run
        command = f"python -m mordred {self.save_dir+'conf.sdf'} -3 -o {self.save_dir+'Mordred.csv'}"
        # Execute the command
        subprocess.run(command, shell=True, check=True)

        mordred_df = pd.read_csv(self.save_dir + "Mordred.csv")
        mordred_df = mordred_df.drop("name", axis=1)
        mordred_df = pd.concat(
            [self.mord[[self.ID, self.activity_col]], mordred_df], axis=1
        )
        mordred_df.to_csv(f"{self.save_dir}Mordred.csv", index=False)

    def fit(
        self,
        features=[
            "Avalon",
            "RDKit",
            "Circular",
            "MACCS",
            "CATS2D",
            "MAP4",
            "Pharm2DGB",
            "PubChem",
            "SECFP",
            "RDKitDescriptors",
            "MordredDescriptors",
        ],
    ):
        """
        Calculate specified molecular features for the dataset.

        Parameters:
        - features (list, optional): A list of features to calculate. If not provided, all available features will be calculated.

        Returns:
        None
        """
        if features is None:
            # If features list is not provided, calculate all fingerprints and descriptors
            features = [
                "Avalon",
                "RDKit",
                "Circular",
                "MACCS",
                "CATS2D",
                "MAP4",
                "Pharm2DGB",
                "PubChem",
                "SECFP",
                "RDKitDescriptors",
                "MordredDescriptors",
            ]

        if "Avalon" in features:
            self.calculate_avalon_fingerprints()
        if "RDKit" in features:
            self.calculate_rdk_fingerprints()
        if "Circular" in features:
            self.calculate_circular_fingerprints()
        if "MACCS" in features:
            self.calculate_maccs_fingerprints()
        if "CATS2D" in features:
            self.calculate_cats2d_fingerprints()
        if "MAP4" in features:
            self.calculate_map4_fingerprints()
        if "Pharm2DGB" in features:
            self.calculate_pharm2dgb_fingerprints()
        if "PubChem" in features:
            self.calculate_pubchem_fingerprints()
        if "SECFP" in features:
            self.calculate_secfp_fingerprints()
        if "RDKitDescriptors" in features:
            self.calculate_rdk_descriptors()
        if "MordredDescriptors" in features:
            self.calculate_mordred_descriptors()
