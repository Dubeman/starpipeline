import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Add root dir to path
import h5py
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import random
from scipy.ndimage import gaussian_filter
import io
from config import EnvironmentConfig

def get_free_space(path):
    """Get free space in GB"""
    stats = shutil.disk_usage(path)
    return stats.free / (2**30)  # Convert to GB

def estimate_size_per_image(patch_size=63, stride=96):
    """Estimate size in GB for one image"""
    # Calculate approximate patches per image
    avg_image_size = 1024  # Assume average image size
    patches_per_image = ((avg_image_size - patch_size) // stride + 1) ** 2
    # Size per patch (H x W x C x float32)
    patch_size_bytes = patch_size * patch_size * 3 * 4
    # Total for both clean and degraded
    total_bytes = patches_per_image * patch_size_bytes * 2
    return total_bytes / (2**30)  # Convert to GB

def create_degraded_image(image, blur_sigma, noise_sigma, jpg_quality):
    """Apply degradations to image with focus on astronomical imaging characteristics"""
    # Convert to float32
    img = np.array(image).astype(np.float32) / 255.0
    
    # Apply minimal Gaussian blur (for PSF simulation)
    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=[blur_sigma, blur_sigma, 0])
    
    # Apply Gaussian noise (simulates readout noise)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma/255.0, img.shape)
        img = np.clip(img + noise, 0, 1)
    
    # No JPEG compression for astronomical images
    # Just ensure the image is in the correct range
    img = np.clip(img, 0, 1)
    
    return img

def extract_patches(image, patch_size=63, stride=96):
    """Extract patches from image"""
    h, w, c = image.shape
    patches = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)
    
    return np.array(patches)

def generate_training_data(input_dir, output_file, max_size_gb=None, num_patches=None, config=None):
    """Generate training data with degradations"""
    # Check available disk space
    free_space_gb = get_free_space(os.path.dirname(os.path.abspath(output_file)))
    print(f"Available disk space: {free_space_gb:.2f} GB")
    
    if max_size_gb:
        if max_size_gb > free_space_gb:
            raise ValueError(f"Requested size ({max_size_gb}GB) exceeds available space ({free_space_gb:.2f}GB)")
        print(f"Limiting output to {max_size_gb}GB")
    
    # Parameters
    patch_size = 63
    stride = 96
    chunk_size = 500  # Reduced chunk size
    
    # Use astronomical degradation parameters from config
    blur_range = config.ASTRO_BLUR_RANGE
    noise_range = config.ASTRO_NOISE_RANGE
    jpg_quality_range = config.ASTRO_JPG_QUALITY
    
    # Get list of images and estimate total size
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    est_size_per_img = estimate_size_per_image(patch_size, stride)
    est_total_size = est_size_per_img * len(image_files)
    print(f"Estimated total size: {est_total_size:.2f} GB")
    
    if max_size_gb:
        max_images = int(max_size_gb / est_size_per_img)
        image_files = image_files[:max_images]
        print(f"Processing {len(image_files)} images to stay within size limit")
    
    # Initialize HDF5 file
    print(f"Creating output file: {output_file}")
    with h5py.File(output_file, 'w') as f:
        # Create extensible datasets with smaller chunks
        f.create_dataset('data', shape=(0, patch_size, patch_size, 3),
                        maxshape=(None, patch_size, patch_size, 3),
                        chunks=(chunk_size, patch_size, patch_size, 3),
                        dtype=np.float32)
        f.create_dataset('label', shape=(0, patch_size, patch_size, 3),
                        maxshape=(None, patch_size, patch_size, 3),
                        chunks=(chunk_size, patch_size, patch_size, 3),
                        dtype=np.float32)
    
    total_patches = 0
    current_batch_degraded = []
    current_batch_clean = []
    
    print("Generating training data...")
    for img_file in tqdm(image_files):
        try:
            # Load image
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # Process at different scales
            for scale in range(3):
                if scale > 0:
                    scale_factor = 2/(scale + 1)
                    h, w = int(img_np.shape[0] * scale_factor), int(img_np.shape[1] * scale_factor)
                    img_scaled = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_CUBIC)
                else:
                    img_scaled = img_np
                
                # Random degradation parameters
                blur_sigma = np.random.choice(blur_range)
                noise_sigma = np.random.choice(noise_range)
                
                # Apply degradations
                degraded = create_degraded_image(img_scaled, blur_sigma, noise_sigma, jpg_quality_range[0])
                
                # Extract patches
                clean_patch_batch = extract_patches(img_scaled.astype(np.float32) / 255.0, patch_size, stride)
                degraded_patch_batch = extract_patches(degraded, patch_size, stride)
                
                current_batch_degraded.extend(degraded_patch_batch)
                current_batch_clean.extend(clean_patch_batch)
                
                # Save when batch is full
                if len(current_batch_degraded) >= chunk_size:
                    with h5py.File(output_file, 'a') as f:
                        # Convert current batches to arrays
                        degraded_array = np.array(current_batch_degraded)
                        clean_array = np.array(current_batch_clean)
                        
                        # Resize datasets
                        f['data'].resize((total_patches + len(degraded_array), patch_size, patch_size, 3))
                        f['label'].resize((total_patches + len(clean_array), patch_size, patch_size, 3))
                        
                        # Write data
                        f['data'][total_patches:total_patches + len(degraded_array)] = degraded_array
                        f['label'][total_patches:total_patches + len(clean_array)] = clean_array
                        
                        total_patches += len(degraded_array)
                    
                    # Clear the current batches
                    current_batch_degraded = []
                    current_batch_clean = []
                
                if num_patches and total_patches >= num_patches:
                    break
            
            if num_patches and total_patches >= num_patches:
                break
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # Save any remaining patches
    if current_batch_degraded:
        with h5py.File(output_file, 'a') as f:
            degraded_array = np.array(current_batch_degraded)
            clean_array = np.array(current_batch_clean)
            
            f['data'].resize((total_patches + len(degraded_array), patch_size, patch_size, 3))
            f['label'].resize((total_patches + len(clean_array), patch_size, patch_size, 3))
            
            f['data'][total_patches:total_patches + len(degraded_array)] = degraded_array
            f['label'][total_patches:total_patches + len(clean_array)] = clean_array
            
            total_patches += len(degraded_array)
    
    print(f"\nTotal patches generated: {total_patches}")
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate training data for image restoration')
    parser.add_argument('--input_dir', default='DIV2K', help='Input directory containing images')
    parser.add_argument('--output_file', default='train.h5', help='Output HDF5 file')
    parser.add_argument('--num_patches', type=int, default=None, help='Number of patches to generate (optional)')
    parser.add_argument('--max_size_gb', type=float, default=None, help='Maximum size of output dataset in GB (optional)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    config = EnvironmentConfig()
    generate_training_data(args.input_dir, args.output_file, args.max_size_gb, args.num_patches, config)
