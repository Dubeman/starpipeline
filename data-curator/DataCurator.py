import os
import random
from glob import glob
import numpy as np
import logging
from ImageData import ImageData
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from tqdm import tqdm

class DenoisingDataset(Dataset):
    def __init__(self, image_data_list):
        self.image_data_list = image_data_list  # List of ImageData objects

    def __len__(self):
        return len(self.image_data_list)

    def __getitem__(self, idx):
        noisy_images, clean_image = self.image_data_list[idx].to_tensor()
        return noisy_images, clean_image

class DataCurator:
    def __init__(self, dataset_path, output_path, noise_modes=['gaussian', 'poisson', 'blur', 'thermal']):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_data_list = []
        self.noise_modes = noise_modes
        self.noise_lists = {mode: [] for mode in self.noise_modes}
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self):
        start_time = datetime.now()
        sample_size = 100
        # Specifically target the images directory
        images_path = os.path.join(self.dataset_path, 'images', '*.jpg')  # Add 'images' to path
        self.image_files = glob(images_path)[0:sample_size]
        random.shuffle(self.image_files)
        self.image_data_list = []
        
        for image_file in tqdm(self.image_files):
            # Create corresponding label path
            label_file = image_file.replace('images', 'labels').replace('.jpg', '.txt')
            if os.path.exists(label_file):
                image_data = ImageData(image_file, label_file)
                self.image_data_list.append(image_data)  # Only append if label exists
        
        logging.info(f'\033[92mLoaded {len(self.image_data_list)} images with labels in {(datetime.now() - start_time).total_seconds():.2f}s.\033[0m')

    def split_data(self):
        start_time = datetime.now()

        dataset = DenoisingDataset(self.image_data_list)

        # Split into training and validation sets
        train_size = int(0.8 * len(self.image_data_list))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Use DataLoader for batching and shuffling
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        logging.info(f'\033[92mSplit data into training and test sets in {(datetime.now() - start_time).total_seconds():.2f}s.\033[0m')
        return train_loader, val_loader
