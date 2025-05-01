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
import matplotlib.pyplot as plt
import json
import gc
import sys
from PIL import Image  # PIL is generally more memory efficient
print(torch.__version__)
print(torchvision.__version__)


NOISE_ARGS = {
    'gaussian': {'mean': 0, 'var': 0.01, 'clip': True},
    'poisson': {'clip': True},
    'speckle': {'mean': 0, 'var': 0.01, 'clip': True},
    'blur': {'ksize': (5, 5), 'sigmaX': 0}
}
class BoundingBox:
    def __init__(self, class_id, x_center, y_center, width, height):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    @classmethod
    def from_string(cls, label_string):
        parts = label_string.split()
        return cls(int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))

    def to_dict(self):
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_dict(cls, label_dict):
        return cls(label_dict["class_id"], label_dict["x_center"], label_dict["y_center"], label_dict["width"], label_dict["height"])
    
class ImageData:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.image = None
        self.bounding_boxes = []
        self.noise_args = NOISE_ARGS
        self.load_labels()

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = self.image.astype('float32') / 255.0
        return self.image

    def load_labels(self):
        with open(self.label_path, 'r') as file:
            for line in file:
                bounding_box = BoundingBox.from_string(line.strip())
                self.bounding_boxes.append(bounding_box)

    def save_labels(self):
        with open(self.label_path, 'w') as file:
            for bounding_box in self.bounding_boxes:
                file.write(f'{bounding_box.class_id} {bounding_box.x_center} {bounding_box.y_center} {bounding_box.width} {bounding_box.height}\n')

    def add_bounding_box(self, bounding_box):
        self.bounding_boxes.append(bounding_box)

    def remove_bounding_box(self, index):
        del self.bounding_boxes[index]

    def get_bounding_box(self, index):
        return self.bounding_boxes[index]

    def get_bounding_boxes(self):
        return self.bounding_boxes

    def get_image(self):
        if self.image is None:
            return self.load_image()
        return self.image

    def set_image(self, image):
        self.image = image

    def save_image(self):
        io.imsave(self.image_path, self.image)

    def get_image_path(self):
        return self.image_path

    def get_label_path(self):
        return self.label_path

    def get_num_bounding_boxes(self):
        return len(self.bounding_boxes)

    def get_bounding_box_count(self):
        return len(self.bounding_boxes)

    def get_bounding_box_count_for_class(self, class_id):
        return len([bb for bb in self.bounding_boxes if bb.class_id == class_id])
    
    def __repr__(self):
        return f'ImageData(image_path={self.image_path}, label_path={self.label_path}, num_bounding_boxes={len(self.bounding_boxes)})'


 
        


class DataCurationPipeline:
    def __init__(self, dataset_path, output_path, noise_modes=['gaussian', 'poisson', 'blur', 'thermal']):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_data_list = []
        self.noise_modes = noise_modes
        self.noise_lists = {mode: [] for mode in self.noise_modes}
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
        BATCH_SIZE = 5  # Reduced batch size for lower memory usage
        
        for mode in self.noise_modes:
            output_dir = os.path.join(self.output_path, f'{mode}_noise')
            os.makedirs(output_dir, exist_ok=True)

            all_labels = []
            image_names = []
            
            noise_list = self.noise_lists[mode]
            num_batches = (len(noise_list) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, len(noise_list))
                batch = noise_list[start_idx:end_idx]
                
                batch_images = []
                
                for image_data in batch:
                    if not isinstance(image_data, ImageData):
                        continue

                    try:
                        base_name = os.path.basename(image_data.get_image_path())
                        base_name_no_ext = os.path.splitext(base_name)[0]
                        image_names.append(base_name_no_ext)

                        # Load image only when needed
                        image = image_data.get_image()
                        noisy_image = self.inject_noise(image, mode)
                        batch_images.append(noisy_image)

                        labels_dict = {
                            'image_name': base_name_no_ext,
                            'bounding_boxes': [bbox.to_dict() for bbox in image_data.get_bounding_boxes()]
                        }
                        all_labels.append(labels_dict)
                        
                        # Clear memory immediately
                        image_data.image = None
                        del image
                        del noisy_image
                        
                    except Exception as e:
                        logging.error(f"Error processing {base_name}: {str(e)}")
                        continue

                if batch_images:
                    batch_output_path = os.path.join(output_dir, f'{mode}_images_batch_{batch_idx}.npy')
                    np.save(batch_output_path, np.array(batch_images, dtype='float32'))
                    logging.info(f'Saved batch {batch_idx + 1}/{num_batches} with {len(batch_images)} images')
                
                del batch_images
                gc.collect()
                
            labels_output_path = os.path.join(output_dir, f'{mode}_labels.json')
            with open(labels_output_path, 'w') as f:
                json.dump({'image_names': image_names, 'labels': all_labels}, f)

            logging.info(f'Completed processing {len(image_names)} images with {mode} noise')

if __name__ == "__main__":
    # Set up paths
    dataset_path = '/Users/owen/starpipeline/old/starpy/DeepSpaceYoloDataset'
    output_path = '/Users/owen/starpipeline/star-visualizer/DeepSpaceYoloDatasetNoisy'

    # Initialize and run the pipeline
    pipeline = DataCurationPipeline(dataset_path, output_path)
    
    logging.info("Starting data curation pipeline...")
    
    # Load the data
    pipeline.load_data()
    
    # Split the data into different noise groups
    pipeline.split_data()
    
    # Process and save the noisy datasets
    pipeline.save_data()
    
    logging.info("Data curation pipeline completed successfully!")

   
    

    
