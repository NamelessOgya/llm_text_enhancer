import os
import shutil
import unittest
import subprocess

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.config_path = "tests/test_experiments.csv"
        with open(self.config_path, 'w') as f:
            f.write("experiment_id,max_generations,population_size,model_name,task_definition,target_preference\n")
            f.write("test_exp,2,2,gpt-3.5,Test Task Def,Test Pref\n")
            
        self.output_script = "tests/run_test_pipeline.sh"

    def tearDown(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.output_script):
            os.remove(self.output_script)

    def test_generate_pipeline(self):
        # Run generate_pipeline.py
        result = subprocess.run([
            "python3", "src/generate_pipeline.py",
            "--config", self.config_path,
            "--output", self.output_script
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(self.output_script))
        
        with open(self.output_script, 'r') as f:
            content = f.read()
            
        # Check if iteration 0 and 1 are present
        self.assertIn('run_test_pipeline.sh', self.output_script) # just verifying path
        self.assertIn('iter0', content)
        self.assertIn('iter1', content)
        self.assertIn('generate_next_step.sh "test_exp" "0"', content)
        self.assertIn('evaluate_step.sh "test_exp" "0"', content)

if __name__ == '__main__':
    unittest.main()
