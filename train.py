"""
train.py — Bone Fracture Detection Training Script

Trains either a custom CNN or VGG16 transfer learning model on the
Bone Fracture Detection dataset (binary: fractured / not fractured).

Usage:
    python train.py --model cnn --epochs 10 --data_dir ./data
    python train.py --model vgg16 --epochs 3 --data_dir ./data
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import (Conv2D, MaxPool2D, BatchNormalization,
                          Dropout, Flatten, Dense)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(dataset_path: str) -> pd.DataFrame:
    """
    Walk a directory of class-named subdirectories and return a DataFrame
    with columns: image (full path), label (class name).
    """
    images, labels = [], []
    for subfolder in sorted(os.listdir(dataset_path)):
        subfolder_path = os.path.join(dataset_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for fname in os.listdir(subfolder_path):
            if fname.lower().endswith('.jpg'):
                images.append(os.path.join(subfolder_path, fname))
                labels.append(subfolder)
    return pd.DataFrame({'image': images, 'label': labels})


def create_generators(data_dir: str, batch_size: int = BATCH_SIZE):
    """Build train, val, and test ImageDataGenerators from a data directory."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_df = load_data(os.path.join(data_dir, 'train'))
    val_df   = load_data(os.path.join(data_dir, 'val'))
    test_df  = load_data(os.path.join(data_dir, 'test'))

    train_gen = datagen.flow_from_dataframe(
        train_df, x_col='image', y_col='label',
        target_size=IMAGE_SIZE, batch_size=batch_size,
        class_mode='binary', shuffle=True
    )
    val_gen = datagen.flow_from_dataframe(
        val_df, x_col='image', y_col='label',
        target_size=IMAGE_SIZE, batch_size=batch_size,
        class_mode='binary', shuffle=False
    )
    test_gen = datagen.flow_from_dataframe(
        test_df, x_col='image', y_col='label',
        target_size=IMAGE_SIZE, batch_size=batch_size,
        class_mode='binary', shuffle=False
    )
    return train_gen, val_gen, test_gen


# ── Model Builders ────────────────────────────────────────────────────────────
def build_custom_cnn() -> Sequential:
    """
    Custom CNN: 3 convolutional blocks with BatchNorm and Dropout,
    followed by a fully connected classification head.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        BatchNormalization(),
        MaxPool2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ], name='custom_cnn')
    return model


def build_vgg16_model() -> Model:
    """
    VGG16 with frozen ImageNet weights as a feature extractor,
    with a custom binary classification head.
    """
    base = VGG16(weights='imagenet', include_top=False,
                 input_shape=(*IMAGE_SIZE, 3))
    for layer in base.layers:
        layer.trainable = False

    x = Flatten()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inputs=base.input, outputs=out, name='vgg16_transfer')


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print(f"\nLoading data from: {args.data_dir}")
    train_gen, val_gen, test_gen = create_generators(args.data_dir, BATCH_SIZE)

    if args.model == 'cnn':
        model = build_custom_cnn()
        patience = 5
    else:
        model = build_vgg16_model()
        patience = 3

    model.summary()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.SpecificityAtSensitivity(0.5),
                 keras.metrics.AUC()]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.model}_best_model.keras')

    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True,
                        monitor='val_loss', verbose=1),
        EarlyStopping(patience=patience, restore_best_weights=True, verbose=1),
    ]

    print(f"\nTraining {args.model.upper()} for up to {args.epochs} epochs...")
    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    print(f"\nEvaluating on test set...")
    loss, acc, spec, auc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Accuracy:    {acc:.4f}")
    print(f"Test AUC:         {auc:.4f}")
    print(f"Test Specificity: {spec:.4f}")
    print(f"Model saved to:   {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a bone fracture detection model.'
    )
    parser.add_argument(
        '--model', type=str, default='cnn', choices=['cnn', 'vgg16'],
        help='Model architecture: cnn or vgg16 (default: cnn)'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Max training epochs (default: 10)'
    )
    parser.add_argument(
        '--data_dir', type=str, default='./data',
        help='Root directory containing train/val/test folders (default: ./data)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./models',
        help='Directory to save trained models (default: ./models)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
