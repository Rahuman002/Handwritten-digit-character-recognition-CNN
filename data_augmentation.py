"""
data_augmentation.py – Apply random augmentations to handwritten images.
Use this during training to improve generalization.
"""

import cv2
import numpy as np
import random

def augment_image(image, 
                  rotation_range=5,
                  shift_range=0.05,
                  zoom_range=0.05,
                  brightness_range=(0.9, 1.1),
                  blur_prob=0.2):
    """
    image: grayscale numpy array (H, W)
    returns: augmented image
    """
    h, w = image.shape
    
    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # Random shift
    if random.random() < 0.5:
        dx = int(random.uniform(-shift_range, shift_range) * w)
        dy = int(random.uniform(-shift_range, shift_range) * h)
        M = np.float32([[1,0,dx], [0,1,dy]])
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # Random zoom (crop + resize)
    if random.random() < 0.5:
        zoom = random.uniform(1-zoom_range, 1+zoom_range)
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        image = cv2.resize(image, (new_w, new_h))
        # Pad or crop back to original size
        if new_w >= w:
            start_x = (new_w - w) // 2
            image = image[:, start_x:start_x+w]
        else:
            pad_x = (w - new_w) // 2
            image = np.pad(image, ((0,0), (pad_x, pad_x)), mode='constant', constant_values=255)
        if new_h >= h:
            start_y = (new_h - h) // 2
            image = image[start_y:start_y+h, :]
        else:
            pad_y = (h - new_h) // 2
            image = np.pad(image, ((pad_y, pad_y), (0,0)), mode='constant', constant_values=255)
    
    # Random brightness (by scaling pixel values)
    if random.random() < 0.5:
        brightness = random.uniform(*brightness_range)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    # Gaussian blur
    if random.random() < blur_prob:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    return image

# Example usage
if __name__ == "__main__":
    img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
    aug = augment_image(img)
    cv2.imwrite("augmented.png", aug)
    print("Augmented image saved.")