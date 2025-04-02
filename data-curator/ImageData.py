import os
from skimage import io
import cv2
import numpy as np
from BoundingBox import BoundingBox
from parameters import NOISE_ARGS, BATCH_SIZE
from parameters import output_path
import torch

def inject_noise(image: np.ndarray, mode: str, iteration: int) -> np.ndarray:
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

    img = noisy_image.astype('float32')  

    return NoisedImage(img, mode, iteration)

class NoisedImage:
    def __init__(self, image, noise_mode="gaussian", iteration=0):
        self.image = image
        self.noise_mode = noise_mode
        self.iteration = iteration

class ImageData:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.name = os.path.basename(image_path).split(".")[0]
        self.label_path = label_path
        self.image = self.load_image()
        self.bounding_boxes = []
        self.noise_args = NOISE_ARGS
        self.load_labels()
        self.variations = []
        self.generate_variations()
       

        #self.image = None
        #del self.image
        #del self.variations
    
    def generate_variations(self):
        for i in range(BATCH_SIZE):
            for mode in self.noise_args:
                self.variations.append(inject_noise(self.image, mode, i))

        #self.save_variations()
    
    def save_variations(self):
        for i, variation in enumerate(self.variations):
            if os.path.exists(os.path.join(output_path, self.name)) == False:
                os.makedirs(os.path.join(output_path, self.name))
            output = os.path.join(output_path, f"{self.name}/{variation.noise_mode}_{variation.iteration}.npy")
            np.save(output, np.array(variation.image, dtype='float32'))


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

    def to_tensor(self):
        noisy_images = [torch.tensor(variation.image, dtype=torch.float32).permute(2, 0, 1) for variation in self.variations]  # List of noisy images (C, H, W)
        clean_image = torch.tensor(self.image, dtype=torch.float32).permute(2, 0, 1)  # Clean image (C, H, W)
        return noisy_images, clean_image