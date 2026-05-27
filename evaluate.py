# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import glob

# -------------------------------
# 1. Configuration
# -------------------------------
DATA_DIR = r"C:\Users\CRESCENT\Desktop\Handwritten\input"
MODEL_SAVE_PATH = "handwriting_model.h5"
WEIGHTS_SAVE_PATH = "model_weights.weights.h5"

IMG_HEIGHT = 64
IMG_WIDTH = 800
MAX_TEXT_LEN = 100   # Global maximum text length (pad/truncate)
BATCH_SIZE = 16
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0005

# Character set (lowercase, digits, punctuation, space)
characters = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()"
num_classes = len(characters) + 1  # +1 for CTC blank (last index)

# Map characters to indices 0..num_classes-2 (blank is last index)
char_to_idx = {c: i for i, c in enumerate(characters)}   # indices 0 to 48
idx_to_char = {i: c for i, c in enumerate(characters)}   # same
blank_index = num_classes - 1   # = 49

print(f"Number of classes (including blank at index {blank_index}): {num_classes}")
print(f"Character set: {characters}")

# -------------------------------
# 2. Data Loading
# -------------------------------
def clean_text(text):
    text = text.lower()
    cleaned = ''.join([c for c in text if c in char_to_idx])
    return cleaned

def load_dataset(data_dir):
    images = []
    texts = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        for img_path in glob.glob(os.path.join(data_dir, '**', ext), recursive=True):
            img_name = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_name)[0]
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read().strip()
            else:
                parts = img_name_no_ext.split('_')
                if len(parts) > 2:
                    raw_text = parts[-1]
                else:
                    continue
            text = clean_text(raw_text)
            if not text:
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Optionally crop/resize to save memory? We'll resize later in generator.
            images.append(img)
            texts.append(text)
    return images, texts

def preprocess_image(image, target_height=IMG_HEIGHT, target_width=IMG_WIDTH):
    h, w = image.shape
    new_w = int(w * target_height / h)
    resized = cv2.resize(image, (new_w, target_height))
    if new_w >= target_width:
        processed = resized[:, :target_width]
    else:
        pad_width = target_width - new_w
        processed = np.pad(resized, ((0,0), (0, pad_width)), mode='constant', constant_values=255)
    processed = processed.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=-1)
    return processed

def encode_text(text):
    return [char_to_idx[c] for c in text]

def decode_text(indices):
    return ''.join([idx_to_char.get(idx, '') for idx in indices if idx != blank_index])

def data_generator(images, texts, batch_size=BATCH_SIZE):
    num_samples = len(images)
    fixed_input_length = IMG_WIDTH // 8  # = 100
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_images = []
            batch_labels = np.full((len(batch_indices), MAX_TEXT_LEN), blank_index, dtype=np.int32)
            batch_label_lengths = []
            for i, idx in enumerate(batch_indices):
                img = preprocess_image(images[idx])
                batch_images.append(img)
                encoded = encode_text(texts[idx])
                length = min(len(encoded), MAX_TEXT_LEN)
                batch_labels[i, :length] = encoded[:length]
                batch_label_lengths.append(length)
            batch_images = np.array(batch_images)
            input_length = np.full((len(batch_indices), 1), fixed_input_length, dtype=np.int32)
            label_length = np.array(batch_label_lengths, dtype=np.int32).reshape(-1, 1)
            yield {
                'image_input': batch_images,
                'labels': batch_labels,
                'input_length': input_length,
                'label_length': label_length
            }, np.zeros(len(batch_indices))

# -------------------------------
# 3. CRNN Model
# -------------------------------
def build_crnn():
    input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='image_input')
    labels = layers.Input(name='labels', shape=[MAX_TEXT_LEN], dtype='int32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int32')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int32')
    
    # CNN
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(input_img)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2,1))(x)
    x = layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D((2,1))(x)
    
    # Reshape
    shape = x.shape
    x = layers.Reshape((shape[2], shape[3] * shape[1]))(x)
    
    # RNN
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2))(x)
    
    # Output
    y_pred = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # CTC loss
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    loss_out = layers.Lambda(ctc_lambda_func, name='ctc')([y_pred, labels, input_length, label_length])
    
    model = models.Model(inputs=[input_img, labels, input_length, label_length], outputs=loss_out)
    inference_model = models.Model(inputs=input_img, outputs=y_pred)
    return model, inference_model

def ctc_loss(y_true, y_pred):
    return y_pred

def train_model(model, train_gen, val_gen, steps_per_epoch, validation_steps):
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=ctc_loss)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(WEIGHTS_SAVE_PATH, save_weights_only=True,
                                           save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]
    history = model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                        validation_data=val_gen, validation_steps=validation_steps,
                        epochs=EPOCHS, callbacks=callbacks, verbose=1)
    return history

def plot_training_history(history):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()

# -------------------------------
# 4. Main Execution
# -------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Handwritten Text Recognition - Training Script")
    print("="*60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found: {DATA_DIR}")
        sys.exit(1)
    
    images, texts = load_dataset(DATA_DIR)
    print(f"Loaded {len(images)} images with transcriptions")
    if len(images) == 0:
        print("No valid images found.")
        sys.exit(1)
    
    text_lengths = [len(t) for t in texts]
    print(f"Avg length: {np.mean(text_lengths):.2f}, Min: {np.min(text_lengths)}, Max: {np.max(text_lengths)}")
    
    train_images, val_images, train_texts, val_texts = train_test_split(
        images, texts, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    steps_per_epoch = max(1, len(train_images) // BATCH_SIZE)
    validation_steps = max(1, len(val_images) // BATCH_SIZE)
    
    train_gen = data_generator(train_images, train_texts, BATCH_SIZE)
    val_gen = data_generator(val_images, val_texts, BATCH_SIZE)
    
    model, inference_model = build_crnn()
    model.summary()
    
    print("\nStarting training...")
    history = train_model(model, train_gen, val_gen, steps_per_epoch, validation_steps)
    
    inference_model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")
    plot_training_history(history)
    
    # Test on a sample
    sample_idx = random.randint(0, len(val_images)-1)
    sample_img = preprocess_image(val_images[sample_idx])
    sample_batch = np.expand_dims(sample_img, axis=0)
    y_pred = inference_model.predict(sample_batch, verbose=0)
    input_len = tf.ones(tf.shape(y_pred)[0]) * tf.shape(y_pred)[1]
    decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(y_pred, perm=[1,0,2]), tf.cast(input_len, tf.int32))
    pred_text = decode_text(decoded[0].values.numpy())
    print(f"Ground truth: {val_texts[sample_idx]}")
    print(f"Prediction:   {pred_text}")
    print("="*60)