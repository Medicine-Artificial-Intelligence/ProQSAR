import unittest
from ProQSAR.Config.config import Config
from ProQSAR.Rescaler.rescaler import Rescaler


class TestConfig(unittest.TestCase):
	def test_default_instances_created(self):
		cfg = Config()
		# spot check a few components
		self.assertIsNotNone(cfg.rescaler)
		self.assertIsInstance(cfg.rescaler, Rescaler)

	def test_param_dict_applies_settings(self):
		cfg = Config(rescaler={"select_method": "StandardScaler", "save_method": True})
		self.assertEqual(cfg.rescaler.select_method, "StandardScaler")
		self.assertTrue(cfg.rescaler.save_method)

	def test_tuple_or_list_converted_to_dict(self):
		cfg = Config(rescaler=[("select_method", "RobustScaler")])
		self.assertEqual(cfg.rescaler.select_method, "RobustScaler")

	def test_existing_instance_is_passed_through(self):
		r = Rescaler(select_method="StandardScaler")
		cfg = Config(rescaler=r)
		self.assertIs(cfg.rescaler, r)


if __name__ == "__main__":
	unittest.main() 