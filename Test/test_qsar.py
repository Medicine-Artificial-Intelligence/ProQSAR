import os
import unittest
import pandas as pd
from tempfile import TemporaryDirectory
from ProQSAR.Config.config import Config
from ProQSAR.qsar import ProQSAR

class TestProQSAR(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        # load and set up data
        self.df = pd.read_csv('Data/testcase.csv')
        self.df_train = self.df.sample(n=40, random_state=42)
        self.df_train.reset_index(inplace=True, drop=True)
        self.df_pred = self.df.drop(self.df_train.index)
        self.df_pred.reset_index(inplace=True, drop=True)

        # set up config
        self.temp_dir = TemporaryDirectory()
        self.cfg = Config(featurizer={'feature_types': ['ECFP4','RDK5']})
        self.qsar = ProQSAR(
            activity_col='pChEMBL',
            id_col='Smiles',
            smiles_col='Smiles',
            config=self.cfg,
            project_name=self.temp_dir.name,
        )

    def tearDown(self):
        """
        Clean up the test environment.
        """
        self.temp_dir.cleanup()

    def test_run_all(self):
        """
        Test the run method.
        """
        self.qsar.run_all(data_dev=self.df_train, data_pred=self.df_pred, alpha=[0.05, 0.1])
        self.assertIsNotNone(self.qsar.model_dev.model)
        #self.assertTrue(os.path.isfile(os.path.join(self.temp_dir.name, "cv_report_model.csv")))
        self.assertIsInstance(self.qsar.model_dev.report, pd.DataFrame)
        self.assertIsInstance(self.qsar.selected_feature, str)
        self.assertIsInstance(self.qsar.shape_summary, dict)

    # def test_get_params(self):
    #     """
    #     Test the get_params method.
    #     """
    #     params = self.qsar.get_params()
    #     self.assertIsNotNone(params)
    #     self.assertIsInstance(params, dict)
    #     self.assertIn("activity_col", params)
    #     self.assertIn("id_col", params)
    #     self.assertIn("smiles_col", params)
    #     self.assertIn("config", params)
    
