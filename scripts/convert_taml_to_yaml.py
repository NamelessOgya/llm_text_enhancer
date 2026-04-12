import os
import glob

def convert_taml_to_yaml(taml_path):
    yaml_path = taml_path[:-5] + ".yaml"
    with open(taml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    yaml_lines = []
    current_section = None
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1]
            # YAML format requires section name as key, and block scalar
            yaml_lines.append(f"{current_section}: |-\n")
        else:
            if current_section:
                # Add 2 spaces for indentation
                yaml_lines.append("  " + line)
                
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.writelines(yaml_lines)
        
    os.remove(taml_path)

if __name__ == "__main__":
    # find all .taml files in the current workspace recursively
    taml_files = glob.glob("**/*.taml", recursive=True)
    count = 0
    for file_path in taml_files:
        print(f"Converting {file_path} to YAML...")
        convert_taml_to_yaml(file_path)
        count += 1
    print(f"Successfully converted {count} files.")
