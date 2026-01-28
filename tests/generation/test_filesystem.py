import unittest
import os
import shutil
import tempfile
import argparse
from generation.filesystem import setup_row_directories, save_generation_results

class TestFilesystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_setup_row_directories(self):
        dirs = setup_row_directories(self.test_dir, 0, 1)
        self.assertTrue(os.path.exists(dirs["prompts_dir"]))
        self.assertTrue(os.path.exists(dirs["row_dir"]))
        
    def test_save_generation_results(self):
        dirs = setup_row_directories(self.test_dir, 0, 1)
        outputs = [("text", "logic")]
        save_generation_results(outputs, dirs)
        
        self.assertTrue(os.path.exists(os.path.join(dirs["texts_dir"], "text_0.txt")))
        self.assertTrue(os.path.exists(os.path.join(dirs["logic_dir"], "creation_prompt_0.txt")))
