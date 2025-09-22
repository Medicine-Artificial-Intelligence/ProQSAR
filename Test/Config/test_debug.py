import unittest
import os
import tempfile
import logging
from proqsar.Config.debug import setup_logging


class TestDebugSetupLogging(unittest.TestCase):
    def test_setup_logging_console(self):
        logger = setup_logging("DEBUG")
        self.assertIsInstance(logger, logging.Logger)
        self.assertGreaterEqual(logger.level, logging.DEBUG)

    def test_setup_logging_invalid_level(self):
        with self.assertRaises(ValueError):
            setup_logging("INVALIDLEVEL")

    def test_setup_logging_file(self):
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "logs", "out.log")
            _ = setup_logging("INFO", log_filename=log_path)
            self.assertTrue(os.path.exists(log_path))


if __name__ == "__main__":
    unittest.main()
