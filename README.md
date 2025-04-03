# Bone_Fracture_classification
Bone Fracture Classification with CNN and VGG16

**Overview**

This deep learning project classifies bone fractures in radiology X-ray images to improve diagnostic accuracy in orthopedics. Using a Kaggle dataset of 17,000 images, I developed a custom Convolutional Neural Network (CNN) and fine-tuned the pre-trained VGG16 model, achieving 92% accuracy and an F1-score of 0.928. This work highlights my skills in medical image analysis, deep learning, and model optimization using Python and Keras


**Objectives**

Detect and classify fractured vs. non-fractured bones in X-ray images.

Compare a custom CNN with the VGG16 model for performance and efficiency.

Leverage preprocessing and deep learning to enhance classification accuracy.


**Dataset**

Source: Bone Fracture dataset from Kaggle.

Size: 17,000 X-ray images (13,000 train, 4,000 test).

Features: Grayscale X-ray images labeled as fractured or non-fractured.

Challenges: Variable resolutions (500–600 pixels), requiring preprocessing.


**Methodology**

1. **Data Preprocessing**
   
Converted images to grayscale for computational efficiency.

Resized images to 224x224x3 pixels using Keras ImageDataGenerator.

Applied augmentation (e.g., scaling, rotation) to improve model generalization.


2.**Custom CNN Model**

Designed with Keras Sequential API for binary classification:

Three 2D convolutional layers (32 filters, 3x3 kernel, ReLU activation).

Batch normalization for training stability.

Max-pooling (2x2) for down-sampling.

Dropout (0.3) to reduce overfitting.

Dense layers (256, 128 neurons) and a final sigmoid layer.

Compiled with Adam optimizer, binary cross-entropy loss, and early stopping (patience=5)

3.**VGG16 Model**

Fine-tuned pre-trained VGG16 (ImageNet weights):

Froze convolutional base, added custom classification layers.

Used Adam optimizer, binary cross-entropy loss, and early stopping (patience=3).

Saved best weights with ModelCheckpoint


**Evaluation**

Metrics: Accuracy, F1-score, specificity, sensitivity, confusion matrix.

Training: 10 epochs (CNN), 3 epochs (VGG16), batch size of 32.


**Key Insights**

Accuracy: Both models achieved 92% test accuracy and an F1-score of 0.928.

Efficiency: VGG16 converged in 3 epochs vs. 10 for the custom CNN, showcasing transfer learning’s advantage.

Performance: High specificity and sensitivity confirmed robust fracture detection, supported by confusion matrix analysis.


**Results**
Custom CNN: 99% accuracy with a tailored architecture.

VGG16: 99% accuracy in fewer epochs, leveraging pre-trained features.

Visualizations: Accuracy/loss curves and confusion matrices included in the notebook.


**Skills Demonstrated**

Deep Learning: Built and fine-tuned CNN and VGG16 models for image classification.

Image Preprocessing: Resized and augmented X-ray images for deep learning.

Model Optimization: Used regularization, early stopping, and transfer learning.

Tools: Python, TensorFlow, Keras, Matplotlib, Seaborn.


**Future Enhancements**

Experiment with hybrid CNN-VGG16 architectures.

Incorporate additional imaging modalities (e.g., CT scans).

Explore advanced models like ResNet50 or YOLO.

Author
Mohammed Zia Sultan
