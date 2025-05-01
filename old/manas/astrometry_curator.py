import os
import cv2
import numpy as np
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen, PIPE
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base command configuration from your astrometry_handler
BASE_CMD = [
    "solve-field", 
    "--overwrite", 
    "--no-plots",
    "--new-fits", "none",
    "--solved", "none",
    "--match", "none",
    "--rdls", "none",
    "--index-xyls", "none",
    "--keep-xylist", "none",
    "--wcs", "none",
    "--corr", "none",
    "--temp-axy",
    "--uniformize", "0",
    "--no-remove-lines",
    "--no-background-subtraction"
]

def build_solve_command(image_path):
    """Build the solve-field command with appropriate parameters"""
    cmd = BASE_CMD.copy()
    
    # Add scale parameters
    cmd.extend(["--scale-units", "arcsecperpix", 
                "--scale-low", "0.1", 
                "--scale-high", "180"])
    
    # Add CPU limit
    cmd.extend(["--cpulimit", "30"])
    
    # Add image path
    cmd.append(image_path)
    
    return cmd

def run_solver(cmd):
    """Run the astrometry solver and extract coordinates"""
    try:
        solve = Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)
        
        for line in solve.stdout:
            if "RA,Dec = (" in line:
                coords = extract_coordinates(line)
                if coords is not None:
                    return True, coords
        solve.wait()
        return False, None
    except Exception as e:
        logger.error(f"Solver error: {str(e)}")
        return False, None

def extract_coordinates(line):
    """Extract coordinates from solver output"""
    pattern = r'\((\d+\.\d+),(\d+\.\d+)\)'
    match = re.search(pattern, line)
    if match:
        ra, dec = map(float, match.groups())
        return (ra, dec)
    return None

def solve_single_image(image_array, temp_dir, idx, is_noised=False):
    """Process single image using solve-field"""
    image = (image_array * 255).astype(np.uint8)
    img_type = 'noised' if is_noised else 'original'
    temp_path = os.path.join(temp_dir, f'{img_type}_{idx}.jpg')
    
    try:
        # Save image
        cv2.imwrite(temp_path, image)
        
        # Build and run solver command
        cmd = build_solve_command(temp_path)
        solved, coords = run_solver(cmd)
        
        if solved:
            logger.info(f"Successfully solved {img_type} image {idx} - Coordinates: {coords}")
        else:
            logger.warning(f"Failed to solve {img_type} image {idx}")
            
    except Exception as e:
        logger.error(f"Error processing {img_type} image {idx}: {str(e)}")
        solved = False
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return {
        'index': idx,
        'type': img_type,
        'solved': solved,
        'coordinates': coords if solved else None
    }

def process_batch(batch_path, max_workers=4):
    """Process a complete batch of images"""
    logger.info(f"Processing batch: {batch_path}")
    
    try:
        # Load the batch
        data = np.load(batch_path)
        logger.info(f"Loaded batch with shape: {data.shape}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            
            # Process images in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit original images
                original_futures = [
                    executor.submit(solve_single_image, data[0, idx], temp_dir, idx, False)
                    for idx in range(data.shape[1])
                ]
                
                # Submit noised images
                noised_futures = [
                    executor.submit(solve_single_image, data[1, idx], temp_dir, idx, True)
                    for idx in range(data.shape[1])
                ]
                
                # Collect results
                for future in original_futures + noised_futures:
                    results.append(future.result())
        
        # Analyze results
        analyze_results(results, data.shape[1])
        return results
    
    except Exception as e:
        logger.error(f"Error processing batch {batch_path}: {str(e)}")
        return []

def analyze_results(results, total_images):
    """Analyze and print solving statistics"""
    original_solved = sum(1 for r in results if not r['type'] == 'noised' and r['solved'])
    noised_solved = sum(1 for r in results if r['type'] == 'noised' and r['solved'])
    
    logger.info("\nSolver Results:")
    logger.info(f"Original images solved: {original_solved}/{total_images} ({original_solved/total_images*100:.1f}%)")
    logger.info(f"Noised images solved: {noised_solved}/{total_images} ({noised_solved/total_images*100:.1f}%)")

if __name__ == "__main__":
    # Test with a single batch
    test_batch = "/Users/manasdubey2022/Desktop/SWEN 711 RL/starpipeline/old/manas/Datasets/DeepSpaceYoloDatasetNoisy/gaussian_noise/gaussian_images_batch_0.npy"
    logger.info("Starting single batch test...")
    process_batch(test_batch)