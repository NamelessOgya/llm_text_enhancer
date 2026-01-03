import unittest
import os
import shutil
import subprocess
import csv
import glob
import json

class TestIntegrationFlow(unittest.TestCase):
    def setUp(self):
        # Setup paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.project_root, "config")
        self.result_dir = os.path.join(self.project_root, "result")
        
        # Create a temporary config for integration test
        self.test_config_path = os.path.join(self.config_dir, "test_integration.csv")
        self.experiment_id = "integration_test_run"
        self.pipeline_script = os.path.join(self.project_root, "run_integration_test.sh")
        
        with open(self.test_config_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["experiment_id", "max_generations", "population_size", "model_name", "adapter_type", "evaluator_type", "task_definition", "target_preference"])
            writer.writerow([self.experiment_id, "2", "3", "dummy", "dummy", "rule_keyword", "Generate valid Pokemon", "Fire Type"])

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.pipeline_script):
            os.remove(self.pipeline_script)
        
        # Cleanup results for this test
        exp_result_dir = os.path.join(self.result_dir, self.experiment_id)
        if os.path.exists(exp_result_dir):
            shutil.rmtree(exp_result_dir)

    def test_end_to_end_pipeline(self):
        # 1. Generate Pipeline Script
        gen_cmd = [
            "python3", os.path.join(self.project_root, "src", "generate_pipeline.py"),
            "--config", self.test_config_path,
            "--output", self.pipeline_script
        ]
        result = subprocess.run(gen_cmd, cwd=self.project_root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Pipeline generation failed: {result.stderr}")
        self.assertTrue(os.path.exists(self.pipeline_script))

        # 2. Make executable
        os.chmod(self.pipeline_script, 0o755)

        # 3. Run Pipeline Script
        # We need to ensure we run it from project root so it finds ./cmd/...
        run_cmd = [self.pipeline_script]
        result = subprocess.run(run_cmd, cwd=self.project_root, capture_output=True, text=True)
        
        # Print output for debugging if it fails
        if(result.returncode != 0):
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        self.assertEqual(result.returncode, 0, f"Pipeline execution failed: {result.stderr}")

        # 4. Verify Outputs
        # Check Iteration 0
        iter0_dir = os.path.join(self.result_dir, self.experiment_id, "iter0")
        metrics_file = os.path.join(iter0_dir, "metrics.json")
        self.assertTrue(os.path.exists(metrics_file))
        self.assertTrue(len(glob.glob(os.path.join(iter0_dir, "texts", "*.txt"))) > 0)
        
        # Verify content of metrics.json contains 'reason'
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            self.assertTrue(len(metrics) > 0)
            self.assertIn("reason", metrics[0])
            self.assertIn("score", metrics[0])
        
        # Check Iteration 1
        iter1_dir = os.path.join(self.result_dir, self.experiment_id, "iter1")
        self.assertTrue(os.path.exists(os.path.join(iter1_dir, "metrics.json")))
        self.assertTrue(len(glob.glob(os.path.join(iter1_dir, "texts", "*.txt"))) > 0)
        
        # Check logs
        log_file = os.path.join(self.result_dir, self.experiment_id, "logs", "execution.log")
        self.assertTrue(os.path.exists(log_file))

if __name__ == '__main__':
    unittest.main()
