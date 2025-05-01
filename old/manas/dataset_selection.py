import numpy as np
import os
import shutil

def create_test_dataset(source_dir, dest_dir, num_batches_per_noise=2):
    """
    Create a smaller test dataset by copying a few batches from each noise type
    Args:
        source_dir: Path to original DeepSpaceYoloDatasetNoisy
        dest_dir: Path to create test dataset
        num_batches_per_noise: Number of batch files to copy for each noise type
    """
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process each noise type directory
    for noise_type in ['gaussian_noise', 'poisson_noise', 'blur_noise', 'thermal_noise']:
        source_noise_dir = os.path.join(source_dir, noise_type)
        dest_noise_dir = os.path.join(dest_dir, noise_type)
        
        # Create noise type directory in destination
        os.makedirs(dest_noise_dir, exist_ok=True)
        
        # Get list of batch files
        batch_files = [f for f in os.listdir(source_noise_dir) if f.endswith('.npy')]
        
        # Select first n batch files
        selected_files = batch_files[:num_batches_per_noise]
        
        # Copy selected files
        for file in selected_files:
            shutil.copy2(
                os.path.join(source_noise_dir, file),
                os.path.join(dest_noise_dir, file)
            )
        
        print(f"Copied {len(selected_files)} files for {noise_type}")

# Usage
source_dir = "DeepSpaceYoloDatasetNoisy"
dest_dir = "DeepSpaceYoloDatasetNoisy_test"
create_test_dataset(source_dir, dest_dir, num_batches_per_noise=5)

# Create new tar.gz of test dataset
import tarfile
with tarfile.open("DeepSpaceYoloDatasetNoisy_test.tar.gz", "w:gz") as tar:
    tar.add(dest_dir)

print("Test dataset created and compressed!")
