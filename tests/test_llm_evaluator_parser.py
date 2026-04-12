import unittest
from unittest.mock import MagicMock
from src.evaluator.llm_evaluator import LLMEvaluator

class TestLLMEvaluatorParser(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.evaluator = LLMEvaluator(llm=self.mock_llm)

    def test_parse_valid_json(self):
        response = '{"score": 8, "reason": "Good text."}'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 8.0)
        self.assertEqual(reason, "Good text.")

    def test_parse_valid_json_with_analysis(self):
        # If "reason" is missing but "analysis" exists, it should pick "analysis"
        response = '{"score": 7, "analysis": "Detailed breakdown..."}'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 7.0)
        self.assertEqual(reason, "Detailed breakdown...")

    def test_parse_valid_json_with_both(self):
        # If both exist, "reason" should take priority
        response = '{"score": 9, "analysis": "Detailed breakdown...", "reason": "Great text."}'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 9.0)
        self.assertEqual(reason, "Great text.")

    def test_parse_markdown_wrapped_json(self):
        response = '''```json\n{\n    "score": 6,\n    "reason": "Needs improvement."\n}\n```'''
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 6.0)
        self.assertEqual(reason, "Needs improvement.")

    def test_parse_markdown_wrapped_no_lang(self):
        response = '''```\n{\n    "score": 5,\n    "reason": "Average."\n}\n```'''
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 5.0)
        self.assertEqual(reason, "Average.")

    def test_parse_invalid_json_fallback_quotes(self):
        # JSON is invalid (missing comma, missing curly braces), fallback should kick in
        response = '"score": 4.5 \n "reason": "Bad syntax"'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 4.5)
        self.assertEqual(reason, "Bad syntax")

    def test_parse_invalid_json_fallback_no_quotes(self):
        response = 'Score: 3 \n Reason: Just bad'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 3.0)
        self.assertEqual(reason, "Just bad")

    def test_parse_invalid_json_fallback_with_equals(self):
        response = 'score = 2 \n reason = "Equal sign syntax"'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 2.0)
        self.assertEqual(reason, "Equal sign syntax")

    def test_parse_fallback_bare_numbers(self):
        # Completely devoid of labels
        response = 'I think it deserves a 7.2 because it is nice.'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 7.2)
        self.assertEqual(reason, 'I think it deserves a 7.2 because it is nice.')

    def test_parse_fallback_broken_brackets(self):
        # Common LLM mistake: forgetting closing bracket
        response = '{\n  "score": 9,\n  "reason": "Almost perfect"'
        score, reason = self.evaluator._parse_response(response)
        self.assertEqual(score, 9.0)
        self.assertEqual(reason, "Almost perfect")

if __name__ == '__main__':
    unittest.main()
