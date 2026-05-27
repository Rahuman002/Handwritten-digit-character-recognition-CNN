"""
preprocess_images.py – Resize, normalize, and save preprocessed images.
Useful to speed up training by avoiding on‑the‑fly preprocessing.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_and_save(src_dir, dst_dir, target_height=64, target_width=800):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in tqdm(os.listdir(src_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        new_w = int(w * target_height / h)
        resized = cv2.resize(img, (new_w, target_height))
        if new_w >= target_width:
            processed = resized[:, :target_width]
        else:
            pad_width = target_width - new_w
            processed = np.pad(resized, ((0,0), (0, pad_width)), mode='constant', constant_values=255)
        # Normalize to [0,1] and save as float32 numpy array (or as image)
        norm = processed.astype('float32') / 255.0
        np.save(os.path.join(dst_dir, fname.replace('.png', '.npy').replace('.jpg', '.npy')), norm)
    print(f"Preprocessed images saved to {dst_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python preprocess_images.py <input_folder> <output_folder>")
        sys.exit(1)
    preprocess_and_save(sys.argv[1], sys.argv[2])