import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # Removed Dataset from here
from torchvision import transforms, utils
import random
import math # Added for rotation calculations

# Helper functions for displaying keypoints
def show_keypoints_batch(image_batch, keypoints_batch):
    """Display image with keypoints for a batch."""
    for i in range(image_batch.shape[0]):
        image = image_batch[i].numpy()
        key_pts = keypoints_batch[i].numpy()

        plt.figure(figsize=(2, 2))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
        plt.title(f'Sample {i}')
        plt.axis('off')
    plt.show()

class FacialKeypointsDataset(torch.utils.data.Dataset): # Explicitly reference Dataset
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        # Load image as PIL Image
        image = Image.open(img_name).convert('L') # Convert to grayscale directly
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)

        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


# -- Transforms -- #

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # If image is PIL, convert to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy = image_copy/255.0
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        w, h = image.size # PIL Image dimensions (width, height)

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # Resize PIL Image
        image = image.resize((new_w, new_h), Image.BICUBIC)
        
        # Scale keypoints
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': image, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        w, h = image.size # PIL Image dimensions

        new_w, new_h = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Crop PIL Image
        image = image.crop((left, top, left + new_w, top + new_h))

        # Update keypoints
        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # if image is PIL, convert to numpy first
        if isinstance(image, Image.Image):
            image = np.array(image)

        # if image has 1 channel (grayscale), reshape to (1, H, W)
        if len(image.shape) == 2:
            image = image.reshape(1, image.shape[0], image.shape[1])
        # else, if image has more than 1 channel, change color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        elif len(image.shape) == 3 and image.shape[2] == 3: # RGB
            image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'keypoints': torch.from_numpy(key_pts).type(torch.FloatTensor)}

class RandomRotation(object):
    """Rotate the image and keypoints by a random degree.

    Args:
        degrees (float or tuple): Range of degrees to select from.
            If float, a range (-degrees, +degrees) is used.
            If tuple, a range (min, max) is used.
    """
    def __init__(self, degrees):
        assert isinstance(degrees, (int, float, tuple))
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            assert len(degrees) == 2 and isinstance(degrees[0], (int, float)) and isinstance(degrees[1], (int, float))
            self.degrees = degrees

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        angle = random.uniform(self.degrees[0], self.degrees[1])

        # Rotate PIL image
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # Rotate keypoints
        w, h = image.size
        center_x, center_y = w / 2, h / 2

        # Convert angle to radians for trigonometric functions
        angle_rad = -math.radians(angle) # Negative because PIL rotation is counter-clockwise for positive angle

        # Create rotation matrix
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])

        # Translate keypoints to origin (center of image)
        translated_key_pts = key_pts - np.array([center_x, center_y])

        # Apply rotation
        rotated_key_pts = np.dot(translated_key_pts, rotation_matrix.T) # .T for row vector multiplication

        # Translate keypoints back
        rotated_key_pts = rotated_key_pts + np.array([center_x, center_y])

        return {'image': image, 'keypoints': rotated_key_pts}

class RandomHorizontalFlip(object):
    """
    Horizontally flip the given image and keypoints randomly with a given probability.

    This transform supports:
    - PIL.Image inputs (mode RGB/L/others)
    - NumPy image arrays with shape (H, W) or (H, W, C)
    - Facial keypoints as a NumPy array of shape (N, 2)

    For facial keypoints:
    - x-coordinates are flipped using x' = (w - 1) - x
    - Optionally swaps left/right landmark indices for common 68-point datasets
      (dlib/iBUG format) to ensure semantic consistency (e.g., left eye ↔ right eye).

    Args:
        p (float): probability of applying the flip.
        swap_lr (bool): if True, swaps left/right keypoint indices for 68-point format.
                        Set to False if your dataset already accounts for mirroring or
                        doesn't require index swapping.
        swap_indices (np.ndarray or list): Optional explicit index mapping for swapping.
                        If provided, overrides the default 68-point mapping. It should be
                        a 1-D array/list of length N where new_keypoints[i] = old_keypoints[swap_indices[i]].
                        If None, and swap_lr=True, uses a default for 68 landmarks.

    Returns:
        dict: {'image': <same type as input image>, 'keypoints': np.ndarray (N, 2)}
    """
    def __init__(self, p=0.5, swap_lr=True, swap_indices=None):
        self.p = p
        self.swap_lr = swap_lr
        self.swap_indices = np.array(swap_indices) if swap_indices is not None else None

        # Precompute the default swap mapping for 68 landmarks (dlib/iBUG format, 0-based):
        # 0-16: jaw (no swap needed; mirroring is handled by x-flip)
        # 17-21: right eyebrow  ↔  22-26: left eyebrow
        # 36-41: right eye      ↔  42-47: left eye
        # Nose (27-35), mouth (48-67) generally don't require index swapping—x-flip suffices.
        if self.swap_lr and self.swap_indices is None:
            swap = np.arange(68)
            # Eyebrows
            swap[17:22], swap[22:27] = np.arange(22, 27), np.arange(17, 22)
            # Eyes
            swap[36:42], swap[42:48] = np.arange(42, 48), np.arange(36, 42)
            self.default_68_swap = swap
        else:
            self.default_68_swap = None

    def _flip_pil(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def _flip_numpy_image(self, image_np):
        # image_np shape: (H, W) or (H, W, C)
        return np.fliplr(image_np)

    def _flip_keypoints(self, key_pts, width):
        # key_pts shape: (N, 2), float or int
        flipped = key_pts.copy()
        # x' = (w - 1) - x  (0-based coordinate system)
        flipped[:, 0] = (width - 1) - flipped[:, 0]
        return flipped

    def _maybe_swap_indices(self, key_pts):
        if self.swap_indices is not None:
            if len(self.swap_indices) != key_pts.shape[0]:
                raise ValueError(
                    f"swap_indices length {len(self.swap_indices)} does not match "
                    f"number of keypoints {key_pts.shape[0]}"
                )
            return key_pts[self.swap_indices]
        if self.default_68_swap is not None:
            if key_pts.shape[0] != 68:
                # Dataset size doesn't match 68; skip swapping to avoid corrupting labels.
                return key_pts
            return key_pts[self.default_68_swap]
        return key_pts

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # Input validation
        if not isinstance(key_pts, np.ndarray):
            key_pts = np.asarray(key_pts, dtype=np.float32)
        if key_pts.ndim != 2 or key_pts.shape[1] != 2:
            raise ValueError("keypoints must be a NumPy array of shape (N, 2)")

        # Determine width/height based on image type
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError("NumPy image must have shape (H, W) or (H, W, C)")
            h, w = image.shape[:2]
        else:
            raise TypeError("image must be a PIL.Image or a NumPy ndarray")

        if random.random() < self.p:
            # Flip image
            if isinstance(image, Image.Image):
                image = self._flip_pil(image)
            else:
                image = self._flip_numpy_image(image)

            # Flip keypoints along x
            key_pts = self._flip_keypoints(key_pts, w)

            # Optional left-right semantic swap
            if self.swap_lr or self.swap_indices is not None:
                key_pts = self._maybe_swap_indices(key_pts)

        return {'image': image, 'keypoints': key_pts}
