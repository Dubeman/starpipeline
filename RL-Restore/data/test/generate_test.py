import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Add root dir to path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from config import EnvironmentConfig

def create_degraded_image(image, blur_sigma, noise_sigma):
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
    
    # Ensure the image is in the correct range
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

def generate_test_data(input_dir, output_dir, config=None, num_images=None):
    """Generate test data with degradations"""
    # Create output directories
    moderate_in_dir = os.path.join(output_dir, 'moderate_in')
    moderate_gt_dir = os.path.join(output_dir, 'moderate_gt')
    os.makedirs(moderate_in_dir, exist_ok=True)
    os.makedirs(moderate_gt_dir, exist_ok=True)
    
    # Use astronomical degradation parameters from config
    blur_range = config.ASTRO_BLUR_RANGE
    noise_range = config.ASTRO_NOISE_RANGE
    
    # Parameters
    patch_size = 63
    stride = 96
    
    # Get list of images
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit number of images if specified
    if num_images is not None:
        image_files = image_files[:num_images]
        print(f"Limiting to {num_images} images")
    
    print("Generating test data...")
    for img_file in tqdm(image_files):
        try:
            # Load image
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Process at different scales
            for scale in range(3):
                if scale > 0:
                    scale_factor = 2/(scale + 1)
                    h, w = int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)
                    img_scaled = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                else:
                    img_scaled = img
                
                # Apply degradations
                # Use moderate degradation parameters
                blur_sigma = np.random.choice(blur_range[1:])  # Skip the first (no blur)
                noise_sigma = np.random.choice(noise_range[1:])  # Skip the first (no noise)
                
                degraded = create_degraded_image(img_scaled, blur_sigma, noise_sigma)
                
                # Extract patches
                clean_patches = extract_patches(img_scaled.astype(np.float32) / 255.0, patch_size, stride)
                degraded_patches = extract_patches(degraded, patch_size, stride)
                
                # Save patches
                for i, (clean_patch, degraded_patch) in enumerate(zip(clean_patches, degraded_patches)):
                    # Save ground truth
                    gt_path = os.path.join(moderate_gt_dir, f"{os.path.splitext(img_file)[0]}_scale{scale}_patch{i}.png")
                    clean_uint8 = (clean_patch * 255).astype(np.uint8)
                    cv2.imwrite(gt_path, clean_uint8)
                    
                    # Save degraded
                    in_path = os.path.join(moderate_in_dir, f"{os.path.splitext(img_file)[0]}_scale{scale}_patch{i}.png")
                    degraded_uint8 = (degraded_patch * 255).astype(np.uint8)
                    cv2.imwrite(in_path, degraded_uint8)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate test data for DeepSpace YOLO dataset')
    parser.add_argument('--input_dir', default='data/train/DeepSpaceYoloDataset', 
                       help='Input directory containing DeepSpace YOLO images')
    parser.add_argument('--output_dir', default='data/test', 
                       help='Output directory for test data')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to process (optional)')
    args = parser.parse_args()
    
    # Load configuration
    config = EnvironmentConfig()
    
    # Generate test data
    generate_test_data(args.input_dir, args.output_dir, config, args.num_images) 