import os
import glob
import re

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content.replace("load_taml_sections", "load_yaml")
    new_content = new_content.replace("parse_taml_ref", "parse_yaml_ref")
    new_content = new_content.replace(".taml", ".yaml")
    new_content = new_content.replace("TAML", "YAML")
    new_content = new_content.replace("taml", "yaml")

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

if __name__ == "__main__":
    py_files = glob.glob("src/**/*.py", recursive=True)
    sh_files = glob.glob("*.sh", recursive=True)
    other_files = ["README.md", "SSoT.md"]
    
    all_files = py_files + sh_files + other_files
    for f in all_files:
        if os.path.exists(f):
            replace_in_file(f)
