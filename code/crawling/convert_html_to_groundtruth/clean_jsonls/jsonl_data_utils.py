import json
import os
from pathlib import Path

class JsonlModifier:
    def __init__(self, jsonl_dir_path: Path, new_root_dir: Path):
        self.new_root_dir = Path(new_root_dir)
        self.jsonl_dir_path = Path(jsonl_dir_path)
        
        assert self.new_root_dir.is_dir(), f"`new_root_dir`={self.new_root_dir} is inalid!"
        assert self.jsonl_dir_path.is_dir(), f"`jsonl_dir_path`={self.jsonl_dir_path} is inalid!"
        assert len([f for f in os.listdir(self.jsonl_dir_path) if f.endswith('.jsonl')]) > 0, f"`self.jsonl_dir_path`={self.jsonl_dir_path} exists but no `jsonl` in it!"


    
    def modify_jsonl_paths(self):
        for jsonl_file in self.jsonl_dir_path.glob("*.jsonl"):
            # Skp already `new` files
            if jsonl_file.name.endswith('_new.jsonl'):
                continue

            # Loop lines
            new_lines = []
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    old_path = Path(data.get('path', ''))
                    
                    # Skip files that are not PDFs
                    if 'ipynb_checkpoints' in str(old_path):
                        continue
                    
                    if len(old_path.parts) >= 3:
                        relative_path = Path(*old_path.parts[-3:])
                        new_path = self.new_root_dir / relative_path
                        
                        if not new_path.exists():
                            print(f"ERROR: PDF under path {new_path} does not exist!")
                            return
                        
                        data['path'] = str(new_path)
                        new_lines.append(json.dumps(data) + "\n")
                    else:
                        print(f"ERROR: Cannot parse the old path {old_path}!")
                        return
            
            new_filename = jsonl_file.stem + '_new.jsonl'
            new_jsonl_path = jsonl_file.with_name(new_filename)
            
            # Overwrite the new file even if it already exists
            with open(new_jsonl_path, 'w') as f:
                f.writelines(new_lines)
            
            print(f"Processed file saved to {new_jsonl_path}")
    
    def replace_old_with_new(self):
        for old_file in self.jsonl_dir_path.glob("*.jsonl"):
            if old_file.name.endswith('_new.jsonl'):
                continue  # Skip files that are already '_new' versions
            
            new_file = old_file.with_name(old_file.stem + '_new.jsonl')
            if not new_file.exists():
                print(f"New version of {old_file} does not exist.")
                continue
            
            if self.compare_jsonl_files(old_file, new_file):
                old_file.unlink()  # Delete the old file
                new_file.rename(old_file)  # Rename new file to old file's name
                print(f"Replaced {old_file} with its new version.")
            else:
                print(f"Files {old_file} and {new_file} do not match in non-path data.")
    
    def compare_jsonl_files(self, old_file: Path, new_file: Path) -> bool:
        with open(old_file, 'r') as f_old, open(new_file, 'r') as f_new:
            for old_line, new_line in zip(f_old, f_new):
                old_data = json.loads(old_line)
                new_data = json.loads(new_line)

                # Skip files that are not PDFs
                if 'ipynb_checkpoints' in str(old_data['path']):
                    continue
                
                if {k: v for k, v in old_data.items() if k != 'path'} != {k: v for k, v in new_data.items() if k != 'path'}:
                    return False
        
        return True

        