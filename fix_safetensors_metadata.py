import os
import safetensors
from safetensors.torch import save_file

def fix_safetensors_metadata(model_dir):
    """Add missing format metadata to safetensors files"""
    print(f"Processing safetensors files in {model_dir}")
    
    # Find all safetensors files
    safetensor_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    
    if not safetensor_files:
        print(f"No safetensors files found in {model_dir}")
        return False
    
    print(f"Found {len(safetensor_files)} safetensors files")
    
    success = True
    for file_name in safetensor_files:
        file_path = os.path.join(model_dir, file_name)
        print(f"Processing {file_path}")
        
        try:
            # Create a new path for the fixed file
            fixed_file_path = file_path + ".fixed"
            
            # Extract all tensors from the original file
            tensors = {}
            with safetensors.safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            
            # Save with proper metadata to the new file
            save_file(tensors, fixed_file_path, metadata={'format': 'pt'})
            
            # Replace the original file with the fixed one
            os.replace(fixed_file_path, file_path)
            print(f"✅ Added metadata to {file_path}")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    model_dir = "outputs/models/east_1b/final_model"
    if fix_safetensors_metadata(model_dir):
        print("Successfully fixed metadata in all safetensors files.")
    else:
        print("Failed to fix metadata in some safetensors files. See errors above.")
