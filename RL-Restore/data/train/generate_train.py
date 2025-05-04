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

def process_images(image_files, output_file, config, patch_size=63, stride=96, chunk_size=500):
    total_patches = 0
    current_batch_degraded = []
    current_batch_clean = []
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', shape=(0, patch_size, patch_size, 3),
                        maxshape=(None, patch_size, patch_size, 3),
                        chunks=(chunk_size, patch_size, patch_size, 3),
                        dtype=np.float32)
        f.create_dataset('label', shape=(0, patch_size, patch_size, 3),
                        maxshape=(None, patch_size, patch_size, 3),
                        chunks=(chunk_size, patch_size, patch_size, 3),
                        dtype=np.float32)
    for img_file in tqdm(image_files):
        try:
            img = Image.open(img_file).convert('RGB')
            img_np = np.array(img)
            for scale in range(3):
                if scale > 0:
                    scale_factor = 2/(scale + 1)
                    h, w = int(img_np.shape[0] * scale_factor), int(img_np.shape[1] * scale_factor)
                    img_scaled = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_CUBIC)
                else:
                    img_scaled = img_np
                blur_sigma = np.random.choice(config.ASTRO_BLUR_RANGE)
                noise_sigma = np.random.choice(config.ASTRO_NOISE_RANGE)
                degraded = create_degraded_image(img_scaled, blur_sigma, noise_sigma, config.ASTRO_JPG_QUALITY[0])
                clean_patch_batch = extract_patches(img_scaled.astype(np.float32) / 255.0, patch_size, stride)
                degraded_patch_batch = extract_patches(degraded, patch_size, stride)
                current_batch_degraded.extend(degraded_patch_batch)
                current_batch_clean.extend(clean_patch_batch)
                if len(current_batch_degraded) >= chunk_size:
                    with h5py.File(output_file, 'a') as f:
                        degraded_array = np.array(current_batch_degraded)
                        clean_array = np.array(current_batch_clean)
                        f['data'].resize((total_patches + len(degraded_array), patch_size, patch_size, 3))
                        f['label'].resize((total_patches + len(clean_array), patch_size, patch_size, 3))
                        f['data'][total_patches:total_patches + len(degraded_array)] = degraded_array
                        f['label'][total_patches:total_patches + len(clean_array)] = clean_array
                        total_patches += len(degraded_array)
                    current_batch_degraded = []
                    current_batch_clean = []
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
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
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate train/val split for DeepSpaceYoloDataset')
    parser.add_argument('--input_dir', required=True, help='Input directory containing all images')
    parser.add_argument('--train_output', default='data/train/star_train.h5', help='Output HDF5 file for training')
    parser.add_argument('--val_output', default='data/valid/validation.h5', help='Output HDF5 file for validation')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of images for validation (default 0.2)')
    parser.add_argument('--max_val_images', type=int, default=None, help='Maximum number of validation images to use')
    parser.add_argument('--max_train_images', type=int, default=None, help='Maximum number of training images to use')
    args = parser.parse_args()

    config = EnvironmentConfig()
    all_images = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)
    val_size = int(len(all_images) * args.val_split)
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]

    if args.max_val_images is not None:
        val_images = val_images[:args.max_val_images]
    if args.max_train_images is not None:
        train_images = train_images[:args.max_train_images]

    print(f"Total images: {len(all_images)} | Train: {len(train_images)} | Val: {len(val_images)}")
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_output), exist_ok=True)
    print("Processing training set...")
    process_images(train_images, args.train_output, config)
    print("Processing validation set...")
    process_images(val_images, args.val_output, config)
