import cv2
import numpy as np
import os
from glob import glob
import random
import logging
import gc
from data_curation import ImageData
import json

# Define noise parameters for different types
NOISE_ARGS = {
    'gaussian': {'mean': 0, 'var': 0.05},  # Increased from 0.01 for more visible noise
    'poisson': {'lambda': 0.3},  # Added lambda parameter for stronger Poisson noise
    'blur': {'ksize': (7, 7), 'sigmaX': 2},  # Increased kernel size and added Gaussian blur
    'thermal': {'scale': 0.15}  # Added scale parameter for stronger thermal noise
}


class DataCurationPipelineEncoder:
    def __init__(self, dataset_path, output_path, noise_modes=['gaussian', 'poisson', 'blur', 'thermal']):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_data_list = []
        self.noise_modes = noise_modes
        self.noise_lists = {mode: [] for mode in self.noise_modes}
        self.batch_size = 32
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self):
        # Specifically target the images directory
        images_path = os.path.join(self.dataset_path, 'images', '*.jpg')  # Add 'images' to path
        self.image_files = glob(images_path)
        random.shuffle(self.image_files)
        self.image_data_list = []
        
        for image_file in self.image_files:
            # Create corresponding label path
            label_file = image_file.replace('images', 'labels').replace('.jpg', '.txt')
            if os.path.exists(label_file):
                image_data = ImageData(image_file, label_file)
                self.image_data_list.append(image_data)  # Only append if label exists
        
        logging.info(f'\033[92mLoaded {len(self.image_data_list)} images with labels.\033[0m')

    def split_data(self, num_splits=None):
        if num_splits is None:
            num_splits = len(self.noise_modes)

        num_images = len(self.image_data_list)
        num_per_list = num_images // num_splits
        
        for i in range(num_splits):
            start_idx = i * num_per_list
            end_idx = (i + 1) * num_per_list
            self.noise_lists[self.noise_modes[i % len(self.noise_modes)]].extend(self.image_data_list[start_idx:end_idx])

        remaining_images = self.image_files[num_per_list * num_splits:]
        for i, img in enumerate(remaining_images):
            self.noise_lists[self.noise_modes[i % len(self.noise_modes)]].append(img)

        logging.info(' | '.join([f'{mode.capitalize()}: {len(self.noise_lists[mode])} images' for mode in self.noise_modes]))

    @staticmethod
    def inject_noise(image: np.ndarray, mode: str) -> np.ndarray:
        noise_params = NOISE_ARGS.get(mode, {})
        
        if mode == 'gaussian':
            mean = noise_params.get('mean', 0)
            var = noise_params.get('var', 0.01)
            noise = np.random.normal(mean, np.sqrt(var), image.shape)
            noisy_image = image + noise
            np.clip(noisy_image, 0, 1, out=noisy_image)
        elif mode == 'poisson':
            noisy_image = np.random.poisson(image * 255.0) / 255.0
            np.clip(noisy_image, 0, 1, out=noisy_image)
        elif mode == 'blur':
            ksize = noise_params.get('ksize', (5, 5))
            sigmaX = noise_params.get('sigmaX', 0)
            noisy_image = cv2.GaussianBlur(image, ksize, sigmaX)
        else:
            noisy_image = image.copy()

        return noisy_image.astype('float32')

    def save_data(self):
        
        for mode in self.noise_modes:
            output_dir = os.path.join(self.output_path, f'{mode}_noise')
            os.makedirs(output_dir, exist_ok=True)
            
            noise_list = self.noise_lists[mode]
            num_batches = (len(noise_list) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(noise_list))
                batch = noise_list[start_idx:end_idx]
                
                batch_original_images = []
                batch_noisy_images = []
                
                for image_data in batch:
                    if not isinstance(image_data, ImageData):
                        continue

                    try:
                        # Load original image
                        original_image = image_data.get_image()
                        # Create noisy version
                        noisy_image = self.inject_noise(original_image, mode)
                        
                        batch_original_images.append(original_image)
                        batch_noisy_images.append(noisy_image)
                        
                        # Clear memory immediately
                        image_data.image = None
                        del original_image
                        del noisy_image
                        
                    except Exception as e:
                        logging.error(f"Error processing image: {str(e)}")
                        continue

                if batch_original_images and batch_noisy_images:
                    # Stack original and noisy images together
                    # Shape will be (2, batch_size, 608, 608, 3)
                    combined_batch = np.stack([
                        np.array(batch_original_images, dtype='float32'),
                        np.array(batch_noisy_images, dtype='float32')
                    ])
                    
                    batch_output_path = os.path.join(output_dir, f'{mode}_images_batch_{batch_idx}.npy')
                    np.save(batch_output_path, combined_batch)
                    logging.info(f'Saved batch {batch_idx + 1}/{num_batches} with {len(batch_original_images)} image pairs')
                
                del batch_original_images
                del batch_noisy_images
                gc.collect()

            logging.info(f'Completed processing images with {mode} noise')

if __name__ == "__main__":
    # Set up paths
    dataset_path = '/Users/manasdubey2022/Desktop/SWEN 711 RL/starpipeline/old/starpy/DeepSpaceYoloDataset'
    output_path = './DeepSpaceYoloDatasetNoisy'


    

   
    
    # Initialize and run the pipeline with a small test
    pipeline = DataCurationPipelineEncoder(dataset_path, output_path)
    
    logging.info("Starting full data curation pipeline...")
    
    # Load all data
    pipeline.load_data()
    logging.info(f"Loaded {len(pipeline.image_data_list)} total images")
    
    # Split the data into different noise groups
    pipeline.split_data()
    
    # Process and save the complete noisy datasets
    pipeline.save_data()
    
    # Log completion statistics
    for mode in pipeline.noise_modes:
        output_dir = os.path.join(output_path, f'{mode}_noise')
        num_batches = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        logging.info(f"\nProcessing complete for {mode} noise:")
        logging.info(f"Generated {num_batches} batch files")
        
    logging.info("Full dataset processing completed successfully!")

    
