import unittest
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.factory import get_llm_adapter

class TestGeminiIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure API key is present or skip
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not found in environment variables. Skipping integration test.")

    def test_gemini_generation(self):
        """Test that Gemini adapter can generate text."""
        print("\nTesting Gemini Adapter Connection...")
        
        # Pick the first available model that supports generation
        import google.generativeai as genai
        model_name = "models/gemini-1.5-flash" # fallback
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_name = m.name
                    print(f"Selected model for testing: {model_name}")
                    break
        except Exception as e:
            print(f"Failed to auto-select model, using fallback: {e}")

        adapter = get_llm_adapter("gemini", model_name)
        
        prompt = "Hello, reply with exactly 'Gemini OK'"
        print(f"Input Prompt: {prompt}")
        response = adapter.generate(prompt)
        print(f"Output Response: {response}")
        
        self.assertTrue(len(response) > 0, "Response should not be empty")
        
        # Check token usage
        usage = adapter.get_token_usage()
        print(f"Token Usage: {usage}")
        self.assertGreater(usage["total_tokens"], 0, "Total tokens should be greater than 0")

if __name__ == '__main__':
    unittest.main()
