import unittest
import os
import yaml
import shutil
from unittest.mock import MagicMock, patch
from llm.interface import LLMInterface
from evolution_strategies import EvolutionStrategy, EEDStrategy

class MockLLM(LLMInterface):
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "Default Response"
        
    def get_token_usage(self):
        return {}

class TestEEDFix(unittest.TestCase):
    def setUp(self):
        self.config_dir = "tests/config/task"
        os.makedirs(self.config_dir, exist_ok=True)
        self.task_name = "test_retry_task"
        self.config_path = os.path.join(self.config_dir, f"{self.task_name}.yaml")
        
        # Create dummy config with max_words = 5
        with open(self.config_path, 'w') as f:
            yaml.dump({"max_words": 5}, f)
            
        self.context = {
            "task_name": self.task_name
        }
        self.real_cwd = os.getcwd()

        
    def tearDown(self):
        if os.path.exists("tests/config"):
            shutil.rmtree("tests/config")

    @patch("os.getcwd")
    def test_dynamic_config_loading(self, mock_getcwd):
        target_dir = os.path.join(self.real_cwd, "tests")
        mock_getcwd.return_value = target_dir
        print(f"DEBUG: Mock CWD: {mock_getcwd.return_value}")
        print(f"DEBUG: Config exists? {os.path.exists(self.config_path)}")
        print(f"DEBUG: Config path: {self.config_path}")
        
        strategy = EEDStrategy()
        max_words = strategy._get_max_words(self.context)
        print(f"DEBUG: max_words: {max_words}")
        self.assertEqual(max_words, 5)
        
        # Constraint string check
        constraint = strategy._get_task_constraint(self.context)
        self.assertIn("strictly 5 words", constraint)

    @patch("os.getcwd")
    def test_generate_with_retry_success(self, mock_getcwd):
        mock_getcwd.return_value = os.path.join(self.real_cwd, "tests")
        strategy = EEDStrategy()
        
        # 1st attempt: 10 words (fail)
        # 2nd attempt: 3 words (pass)
        responses = [
            "one two three four five six seven eight nine ten",
            "one two three"
        ]
        llm = MockLLM(responses)
        
        result = strategy._generate_with_retry(llm, "Prompt", self.context)
        
        # Expected result is the second one
        self.assertEqual(result, "one two three")
        
        # Verify calls
        self.assertEqual(len(llm.calls), 2)
        # Second prompt should contain warning
        self.assertIn("Constraint Violation", llm.calls[1])
        self.assertIn("STRICTLY 5 words", llm.calls[1])

    @patch("os.getcwd")
    def test_generate_with_retry_failure(self, mock_getcwd):
        mock_getcwd.return_value = os.path.join(self.real_cwd, "tests")
        strategy = EEDStrategy()
        
        # All attempts fail
        responses = [
            "fail " * 10,
            "fail " * 10,
            "fail " * 10,
            "fail " * 10
        ]
        llm = MockLLM(responses)
        
        # Retries default to 3. So 1 initial + 3 retries = 4 calls total max?
        # Code says: for i in range(max_retries): ...
        # So exactly max_retries attempts. Default is 3.
        
        result = strategy._generate_with_retry(llm, "Prompt", self.context, max_retries=3)
        
        self.assertEqual(len(llm.calls), 3)
        # Should return the last generated text even if failed
        self.assertEqual(result, ("fail " * 10).strip())

if __name__ == "__main__":
    unittest.main()
