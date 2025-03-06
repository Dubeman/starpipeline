import os
import random
from glob import glob
from skimage import io, img_as_float
from skimage.util import random_noise
import cv2
import numpy as np
import logging
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)

class DataCurationPipeline:
    def __init__(self, dataset_path, output_path, noise_modes=['gaussian', 'poisson', 'blur', 'thermal']):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_files = []
        self.noise_modes = noise_modes
        self.noise_lists = {mode: [] for mode in self.noise_modes}
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self):
        self.image_files = glob(os.path.join(self.dataset_path, '*'))
        random.shuffle(self.image_files)
        logging.info(f'\033[92mLoaded {len(self.image_files)} images.\033[0m')

    def split_data(self, num_splits=None):
        if num_splits is None:
            num_splits = len(self.noise_modes)

        num_images = len(self.image_files)
        num_per_list = num_images // num_splits
        
        for i in range(num_splits):
            start_idx = i * num_per_list
            end_idx = (i + 1) * num_per_list
            self.noise_lists[self.noise_modes[i % len(self.noise_modes)]].extend(self.image_files[start_idx:end_idx])

        remaining_images = self.image_files[num_per_list * num_splits:]
        for i, img in enumerate(remaining_images):
            self.noise_lists[self.noise_modes[i % len(self.noise_modes)]].append(img)

        logging.info(' | '.join([f'{mode.capitalize()}: {len(self.noise_lists[mode])} images' for mode in self.noise_modes]))

    def inject_noise(self, image, mode):
        if mode == 'gaussian':
            noisy_image = random_noise(image, mode='gaussian', mean=0, var=0.01)
        elif mode == 'poisson':
            noisy_image = random_noise(image, mode='poisson')
        elif mode == 'blur':
            noisy_image = cv2.GaussianBlur((image * 255).astype(np.uint8), (5, 5), 0)
            noisy_image = img_as_float(noisy_image)
        elif mode == 'thermal':
            thermal_noise = np.random.normal(0, 0.05, image.shape)
            noisy_image = image + thermal_noise
            noisy_image = np.clip(noisy_image, 0, 1)
        else:
            noisy_image = image
        return noisy_image

    def save_data(self):
        for mode in self.noise_modes:
            output_dir = os.path.join(self.output_path, f'{mode}_noise')
            os.makedirs(output_dir, exist_ok=True)
            for img_path in self.noise_lists[mode]:
                image = img_as_float(io.imread(img_path))
                noisy_image = self.inject_noise(image, mode)
                output_file = os.path.join(output_dir, os.path.basename(img_path))
                io.imsave(output_file, noisy_image)
                logging.info(f'Saved {output_file} with {mode} noise.')

if __name__ == "__main__":
    dataset_path = './DeepSpaceYoloDataset/images'
    output_path = './DeepSpaceYoloDatasetNoisy'

    pipeline = DataCurationPipeline(dataset_path, output_path)
    pipeline.load_data()
    pipeline.split_data()
    # pipeline.save_data()
