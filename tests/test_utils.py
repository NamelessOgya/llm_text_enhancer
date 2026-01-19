import unittest
import os
import sys
import shutil
import tempfile

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import load_taml, load_content

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_load_taml_standard(self):
        """Standard usage of TAML with background and content"""
        file_path = os.path.join(self.test_dir, "standard.taml")
        content = """[background]
This is background info.

[content]
This is the content.
Multiline is supported.
"""
        with open(file_path, 'w') as f:
            f.write(content)

        loaded = load_taml(file_path)
        expected = "This is the content.\nMultiline is supported."
        self.assertEqual(loaded, expected)

    def test_load_taml_no_content_tag(self):
        """Fallback when [content] tag is missing (backward compatibility)"""
        file_path = os.path.join(self.test_dir, "plain.taml")
        content = "Just plain text without tags."
        with open(file_path, 'w') as f:
            f.write(content)

        loaded = load_taml(file_path)
        self.assertEqual(loaded, content)

    def test_load_taml_empty_content(self):
        """When [content] tag exists but is empty"""
        file_path = os.path.join(self.test_dir, "empty.taml")
        content = """[background]
info
[content]
"""
        with open(file_path, 'w') as f:
            f.write(content)

        loaded = load_taml(file_path)
        self.assertEqual(loaded, "")

    def test_load_content_with_file(self):
        """load_content should resolve file paths ending in .taml"""
        file_path = os.path.join(self.test_dir, "test.taml")
        with open(file_path, 'w') as f:
            f.write("[content]\nFile Content")
            
        loaded = load_content(file_path)
        self.assertEqual(loaded, "File Content")

    def test_load_content_with_raw_string(self):
        """load_content should return raw string if not a file"""
        raw = "This is raw text"
        # Ensure it doesn't accidentally pick up a file (though unlikely)
        loaded = load_content(raw)
        self.assertEqual(loaded, raw)

    def test_load_taml_not_found(self):
        """load_taml should raise FileNotFoundError if file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_taml("non_existent_file.taml")

if __name__ == '__main__':
    unittest.main()
