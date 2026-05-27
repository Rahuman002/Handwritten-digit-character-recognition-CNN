"""
visualize_predictions.py – Display image and predicted text using matplotlib.
"""

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from model import preprocess_image, decode_text

def visualize(model_path, image_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot read image")
        return
    # Preprocess
    proc = preprocess_image(img)
    y_pred = model.predict(proc, verbose=0)
    input_len = tf.ones(tf.shape(y_pred)[0]) * tf.shape(y_pred)[1]
    decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(y_pred, perm=[1,0,2]), tf.cast(input_len, tf.int32))
    text = decode_text(decoded[0].values.numpy())
    
    # Display
    plt.figure(figsize=(12, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {text}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"Extracted text: {text}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python visualize_predictions.py <model.h5> <image.png>")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2])