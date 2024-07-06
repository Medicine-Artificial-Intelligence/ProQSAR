import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
import pandas as pd
from rdkit import Chem
from AtQSAR.AtCleaning.data_integration import DataIntegration
from AtQSAR.AtCleaning.data_variance import DataVarianceHandler
from AtQSAR.AtCleaning.missing_handling import MissingHandler
from AtQSAR.AtCleaning.multivariate_outliers import MultivariateOutliers
from AtQSAR.AtCleaning.rescale import Rescale
from AtQSAR.AtMetaAnalysis.feature_extraction import FeatureSelectionPipeline
from IPython import display

if __name__ == '__main__':
    df = pd.read_csv(f'{root_dir}/Data/HIV/RDKdes.csv')

    activity_col = 'pChEMBL Value'
    id_col = 'Molecule ChEMBL ID'
    task_type = 'C'
    target_thresh = 8.41
    data_integration = DataIntegration(df, activity_col, id_col, task_type, target_thresh, visualize=False)
    data_train, data_test = data_integration.fit()
    variance = DataVarianceHandler(data_train, data_test, activity_col, id_col, visualize=False)
    data_train, data_test = variance.fit()
    missing = MissingHandler(data_train, data_test, id_col, activity_col)
    data_train, data_test = missing.fit(imputation_strategy='knn', n_neighbors=5)
    

    outliers = MultivariateOutliers(data_train, data_test, method='LocalOutlierFactor', n_jobs=4, 
                                    id_col=id_col, activity_col=activity_col, save_path=None)
    data_train, data_test = outliers.fit()

    
    scale = Rescale(data_train, data_test, id_col, activity_col)
    data_train, data_test = scale.fit()

    fs_pipeline = FeatureSelectionPipeline(data_train, data_test, activity_col, id_col, 'C', 'RF')
    X_train, X_test, y_train, y_test = fs_pipeline.fit()
    print(X_train)
  

