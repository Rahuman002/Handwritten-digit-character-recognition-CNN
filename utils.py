"""
utils.py – Common functions used across other scripts.
"""

import numpy as np
import cv2
import tensorflow as tf

# Character set (must match training)
characters = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()"
num_classes = len(characters) + 1
blank_index = num_classes - 1
idx_to_char = {i: c for i, c in enumerate(characters)}

IMG_HEIGHT = 64
IMG_WIDTH = 800

def preprocess_image(image, target_height=IMG_HEIGHT, target_width=IMG_WIDTH):
    """Preprocess a single image (grayscale, resize, pad, normalize)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape
    new_w = int(w * target_height / h)
    resized = cv2.resize(gray, (new_w, target_height))
    if new_w >= target_width:
        processed = resized[:, :target_width]
    else:
        pad_width = target_width - new_w
        processed = np.pad(resized, ((0,0), (0, pad_width)), mode='constant', constant_values=255)
    processed = processed.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=-1)  # channel
    return processed

def decode_text(indices):
    """Convert CTC output indices to string."""
    return ''.join([idx_to_char.get(idx, '') for idx in indices if idx != blank_index])

def ctc_decode_predictions(y_pred):
    """Decode batch of predictions (numpy array)."""
    pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    pred_tensor = tf.transpose(pred_tensor, perm=[1, 0, 2])
    input_length = tf.ones(tf.shape(pred_tensor)[1], dtype=tf.int32) * tf.shape(pred_tensor)[0]
    decoded, _ = tf.nn.ctc_greedy_decoder(pred_tensor, input_length, merge_repeated=True)
    indices = decoded[0].indices.numpy()
    values = decoded[0].values.numpy()
    text = decode_text(values)
    return text

def load_image_from_path(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img

# Example usage
if __name__ == "__main__":
    print("Utils module loaded. Available functions: preprocess_image, decode_text, ctc_decode_predictions, load_image_from_path")