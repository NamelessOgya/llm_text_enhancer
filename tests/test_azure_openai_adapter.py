import unittest
from unittest.mock import MagicMock, patch
import os
from src.llm.azure_openai_adapter import AzureOpenAIAdapter
from src.llm.factory import get_llm_adapter

class TestAzureOpenAIAdapter(unittest.TestCase):
    def setUp(self):
        # 環境変数の設定 (実際のAPI呼び出しは行わない)
        os.environ["AZURE_OPENAI_API_KEY"] = "fake-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fake-endpoint.openai.azure.com/"

    @patch("openai.AzureOpenAI")
    def test_generate(self, mock_azure_client):
        # モックの設定
        mock_instance = mock_azure_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mocked Response"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_instance.chat.completions.create.return_value = mock_response

        adapter = AzureOpenAIAdapter(model_name="test-deployment")
        result = adapter.generate("Hello", system_prompt="You are a helpful assistant")

        # 検証
        self.assertEqual(result, "Mocked Response")
        self.assertEqual(adapter.get_token_usage()["total_tokens"], 15)
        mock_instance.chat.completions.create.assert_called_once()

    def test_factory_registration(self):
        with patch("src.llm.azure_openai_adapter.AzureOpenAIAdapter.__init__", return_value=None):
            adapter = get_llm_adapter("azure_openai", "test-deployment")
            self.assertIsInstance(adapter, AzureOpenAIAdapter)

if __name__ == "__main__":
    unittest.main()
