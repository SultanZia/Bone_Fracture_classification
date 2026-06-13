"""
predict.py — Bone Fracture Inference with Grad-CAM

Loads a trained model and classifies a single X-ray image as
Fractured or Not Fractured, with an optional Grad-CAM heatmap overlay
showing which regions of the bone influenced the prediction.

Usage:
    python predict.py --model_path models/cnn_best_model.keras --image_path xray.jpg
    python predict.py --model_path models/vgg16_best_model.keras --image_path xray.jpg --gradcam
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
CLASS_NAMES = ['Fractured', 'Not Fractured']

# Last conv layer names for each architecture
LAST_CONV_LAYERS = {
    'custom_cnn':    'conv2d_2',
    'vgg16_transfer': 'block5_conv3',
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize and normalise an X-ray image for inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array: np.ndarray, model: Model,
                          last_conv_layer_name: str) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap highlighting regions that drove the prediction.

    Args:
        img_array:            Preprocessed image (1, H, W, 3)
        model:                Trained Keras model
        last_conv_layer_name: Name of the final convolutional layer

    Returns:
        heatmap: 2D numpy array normalised to [0, 1]
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads  = tape.gradient(class_channel, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image_path: str, heatmap: np.ndarray,
                     alpha: float = 0.4) -> tuple:
    """Overlay a Grad-CAM heatmap on the original image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    heatmap_resized   = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap_coloured  = np.uint8(255 * heatmap_resized)
    heatmap_coloured  = cv2.applyColorMap(heatmap_coloured, cv2.COLORMAP_JET)
    heatmap_coloured  = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_coloured, alpha, 0)
    return img, overlay


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model_path: str, image_path: str, show_gradcam: bool = False):
    """
    Run inference on a single X-ray image.

    Args:
        model_path:   Path to the saved .keras model file.
        image_path:   Path to the X-ray image.
        show_gradcam: If True, display a Grad-CAM overlay.
    """
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Processing image: {image_path}")
    img_array = preprocess_image(image_path)

    prob       = float(model.predict(img_array, verbose=0)[0][0])
    pred_class = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
    confidence = prob if prob >= 0.5 else 1 - prob

    print("\n" + "=" * 45)
    print("  BONE FRACTURE CLASSIFICATION RESULT")
    print("=" * 45)
    print(f"  Prediction : {pred_class}")
    print(f"  Confidence : {confidence:.1%}")
    print(f"  Raw output : {prob:.4f}  (≥0.5 → Not Fractured)")
    print("=" * 45)
    print("\n  ⚠  For clinical use only under radiologist supervision.\n")

    if show_gradcam:
        # Detect architecture from model name
        model_name = model.name
        last_conv  = LAST_CONV_LAYERS.get(model_name)

        if last_conv is None:
            # Fallback: find last Conv2D layer automatically
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break

        if last_conv:
            print(f"Generating Grad-CAM using layer: {last_conv}")
            heatmap = make_gradcam_heatmap(img_array, model, last_conv)
            original, overlay = overlay_gradcam(image_path, heatmap)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Prediction: {pred_class} ({confidence:.1%} confidence)',
                         fontsize=13, fontweight='bold')
            axes[0].imshow(original);   axes[0].set_title('Original X-ray');  axes[0].axis('off')
            axes[1].imshow(overlay);    axes[1].set_title('Grad-CAM Overlay'); axes[1].axis('off')
            plt.tight_layout()
            plt.savefig('gradcam_output.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("Grad-CAM saved to: gradcam_output.png")
        else:
            print("Could not find a Conv2D layer for Grad-CAM.")

    return pred_class, confidence


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify a bone X-ray as Fractured or Not Fractured.'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained .keras model file')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to X-ray image (.jpg or .png)')
    parser.add_argument('--gradcam', action='store_true',
                        help='Show Grad-CAM heatmap overlay')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args.model_path, args.image_path, show_gradcam=args.gradcam)
