import unittest
import os
import shutil
import tempfile
import json
from evaluator_pipeline.filesystem import update_row_score_summary, save_metrics, get_target_content

class TestEvaluationFilesystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_save_metrics(self):
        metrics = [{"score": 1.0}]
        save_metrics(self.test_dir, metrics)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "metrics.json")))
        
    def test_update_row_score_summary(self):
        # Setup iter dir
        iter_dir = os.path.join(self.test_dir, "iter0")
        os.makedirs(iter_dir)
        with open(os.path.join(iter_dir, "metrics.json"), 'w') as f:
            json.dump([{"score": 1.0}, {"score": 2.0}], f)
            
        update_row_score_summary(self.test_dir)
        summary_path = os.path.join(self.test_dir, "score_summary.json")
        self.assertTrue(os.path.exists(summary_path))
        
        with open(summary_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["iter0"]["max"], 2.0)

    def test_get_target_content(self):
        # Mock args
        class Args:
            evaluator_type = "rule_meteor"
        
        row_idx = 0
        dataset = [{"target": "real_answer", "var": "value"}]
        raw_pref = "pref {{ var }}"
        ref_info = {"target": "target"}
        
        # Case 1: Reference from dataset
        target = get_target_content(Args(), row_idx, dataset, raw_pref, ref_info)
        self.assertEqual(target, "real_answer")
        
        # Case 2: No dataset
        dataset = []
        target = get_target_content(Args(), row_idx, dataset, raw_pref, ref_info)
        self.assertEqual(target, "pref {{ var }}") # No replacement if no row data
