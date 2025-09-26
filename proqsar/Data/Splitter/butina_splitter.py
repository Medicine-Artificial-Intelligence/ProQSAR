import pandas as pd
from typing import Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import logging

logger = logging.getLogger(__name__)


class ButinaSplitter:
    """
    Data splitter based on Butina clustering of molecular fingerprints.

    Molecules are clustered using Butina clustering on Tanimoto distances
    of Morgan fingerprints. Clusters are then assigned to train and test
    sets without splitting clusters across sets.

    :param activity_col: Name of the column representing the activity or target label.
    :type activity_col: str
    :param smiles_col: Name of the column containing SMILES strings.
    :type smiles_col: str
    :param mol_col: Name of the column containing RDKit Mol objects (if available).
                    Default is ``"mol"``.
    :type mol_col: str
    :param test_size: Proportion of the dataset to include in the test split.
                      Default is ``0.2``.
    :type test_size: float
    :param cutoff: Tanimoto distance cutoff for clustering (default 0.6).
                   Lower values → finer clusters, higher values → coarser clusters.
    :type cutoff: float

    **Example**

    .. code-block:: python

        import pandas as pd
        from proqsar.Data.Splitter.butina_splitter import ButinaSplitter

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"],
            "activity": [0.1, 0.5, 0.9, 1.2, 1.5]
        })

        splitter = ButinaSplitter(activity_col="activity", smiles_col="SMILES", test_size=0.2)
        train, test = splitter.fit(df)
        print(train.shape, test.shape)
        # e.g., (4, 3) (1, 3)
    """

    def __init__(
        self,
        activity_col: str,
        smiles_col: str,
        mol_col: str = "mol",
        test_size: float = 0.2,
        cutoff: float = 0.6,
    ):
        self.activity_col = activity_col
        self.smiles_col = smiles_col
        self.mol_col = mol_col
        self.test_size = test_size
        self.cutoff = cutoff

    def fit(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training and testing sets using Butina clustering.

        :param data: Input dataset containing SMILES, Mol objects (optional), and activity labels.
        :type data: pd.DataFrame

        :return: A tuple ``(train_df, test_df)`` where
                 - ``train_df`` is the training dataset,
                 - ``test_df`` is the testing dataset.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]

        :raises ImportError: If RDKit is not installed.
        :raises ValueError: If SMILES parsing fails for all molecules.
        """
        # get RDKit mols if not provided
        if self.mol_col not in data.columns:
            data[self.mol_col] = data[self.smiles_col].apply(Chem.MolFromSmiles)

        mols = data[self.mol_col].tolist()
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
            for m in mols
            if m is not None
        ]

        if len(fps) == 0:
            raise ValueError("No valid molecules found for Butina clustering.")

        # compute distance matrix
        dists = []
        nfps = len(fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])

        # cluster molecules
        clusters = Butina.ClusterData(dists, nfps, self.cutoff, isDistData=True)
        clusters = sorted(clusters, key=lambda x: -len(x))  # sort by cluster size

        # allocate clusters into train/test
        frac_train = 1 - self.test_size
        train_cutoff = frac_train * len(data)

        train_idx, test_idx = [], []
        for cluster in clusters:
            if len(train_idx) + len(cluster) > train_cutoff:
                test_idx.extend(cluster)
            else:
                train_idx.extend(cluster)

        assert (
            len(set(train_idx).intersection(set(test_idx))) == 0
        ), "Train and test indices overlap."

        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()

        return train_df, test_df
