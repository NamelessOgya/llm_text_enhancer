import os
import unittest
import tempfile
from src.utils import load_yaml, parse_yaml_ref, load_content

class TestYamlParser(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_temp_yaml(self, content: str) -> str:
        fd, path = tempfile.mkstemp(dir=self.temp_dir.name, suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_load_yaml_basic(self):
        content = """judge_prompt: |-
  Hello world.
  This is a test.
"""
        path = self.create_temp_yaml(content)
        data = load_yaml(path)
        self.assertIn("judge_prompt", data)
        self.assertEqual(data["judge_prompt"], "Hello world.\nThis is a test.")

    def test_load_yaml_special_chars(self):
        # Testing JSON syntax, brackets, colons, hyphens and quotes inside the payload
        content = """content: |-
  You are an evaluator.
  Output JSON format:
  {
      "score": <int>,
      "reason": "String with 'quotes' and \\"double quotes\\""
  }
  
  Guidelines:
  - Do not hallucinate.
  - Pay attention to colons : inside text.
  | Also pipes are fine.
  <thought> Think step by step </thought>
"""
        path = self.create_temp_yaml(content)
        data = load_yaml(path)
        val = data["content"]
        self.assertIn('"score": <int>', val)
        self.assertIn('String with \'quotes\' and \\"double quotes\\"', val)
        self.assertIn("- Do not hallucinate.", val)
        self.assertIn("colons : inside text", val)
        self.assertIn("| Also pipes are fine.", val)
        self.assertIn("<thought> Think step by step </thought>", val)

    def test_load_yaml_multi_section(self):
        content = """background: |-
  Context info.
content: |-
  Main prompt here.
"""
        path = self.create_temp_yaml(content)
        data = load_yaml(path)
        self.assertEqual(data.get("background"), "Context info.")
        self.assertEqual(data.get("content"), "Main prompt here.")

    def test_load_content_with_yaml_path(self):
        # Assuming load_content returns the value of the 'content' key if it's a dict,
        # or the raw string if it has no dict structure.
        # We need to test how load_content adapts to the new load_yaml implementation.
        yaml_text = """content: |-\n  Just the content."""
        path = self.create_temp_yaml(yaml_text)
        result = load_content(path)
        self.assertEqual(result, "Just the content.")

    def test_parse_yaml_ref(self):
        # Formerly parse_taml_ref
        content = """ref:
  dataset: "data/train.csv"
  population: "pop_0"
"""
        path = self.create_temp_yaml(content)
        ref_data = parse_yaml_ref(path)
        self.assertEqual(ref_data.get("dataset"), "data/train.csv")
        self.assertEqual(ref_data.get("population"), "pop_0")

if __name__ == '__main__':
    unittest.main()
