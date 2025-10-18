"""
MNIST Handwritten Digits Classification using CNN
Goal: Achieve >95% test accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ==================== DATA LOADING AND PREPROCESSING ====================
print("\n" + "="*60)
print("LOADING AND PREPROCESSING DATA")
print("="*60)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to add channel dimension (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"\nAfter preprocessing:")
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")

# Convert labels to categorical (one-hot encoding)
num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# ==================== MODEL ARCHITECTURE ====================
print("\n" + "="*60)
print("BUILDING CNN MODEL")
print("="*60)

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build a CNN model for MNIST digit classification
    
    Architecture:
    - Conv2D layers with ReLU activation
    - MaxPooling for dimensionality reduction
    - Dropout for regularization
    - Dense layers for classification
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", 
                     input_shape=input_shape, padding="same", name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", 
                     padding="same", name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(0.25, name="dropout1"),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", 
                     padding="same", name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", 
                     padding="same", name="conv4"),
        layers.BatchNormalization(name="bn4"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="dropout2"),
        
        # Flatten and dense layers
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn5"),
        layers.Dropout(0.5, name="dropout3"),
        layers.Dense(num_classes, activation="softmax", name="output")
    ], name="MNIST_CNN")
    
    return model

# Build the model
model = build_cnn_model()

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully!")
print(f"Total parameters: {model.count_params():,}")

# ==================== TRAINING ====================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
]

# Train the model
batch_size = 128
epochs = 20

history = model.fit(
    x_train, y_train_cat,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")

# ==================== EVALUATION ====================
print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\n{'='*60}")
print(f"TEST RESULTS")
print(f"{'='*60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"{'='*60}")

if test_accuracy > 0.95:
    print(f"✓ SUCCESS: Test accuracy {test_accuracy*100:.2f}% exceeds 95% target!")
else:
    print(f"✗ Test accuracy {test_accuracy*100:.2f}% is below 95% target")

# ==================== VISUALIZATION ====================
print("\n" + "="*60)
print("VISUALIZING PREDICTIONS ON 5 SAMPLE IMAGES")
print("="*60)

# Select 5 random test samples
np.random.seed(42)
sample_indices = np.random.choice(len(x_test), 5, replace=False)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

# Make predictions
predictions = model.predict(sample_images, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

# Create visualization
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle('CNN Predictions on MNIST Test Samples', fontsize=16, fontweight='bold')

for idx, (ax, image, true_label, pred_label, pred_probs) in enumerate(
    zip(axes, sample_images, sample_labels, predicted_labels, predictions)):
    
    # Display image
    ax.imshow(image.squeeze(), cmap='gray')
    
    # Set title with prediction and confidence
    confidence = pred_probs[pred_label] * 100
    color = 'green' if pred_label == true_label else 'red'
    title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'mnist_predictions.png'")

# Display prediction details
print("\nDetailed Predictions:")
print("-" * 60)
for idx, (true_label, pred_label, pred_probs) in enumerate(
    zip(sample_labels, predicted_labels, predictions)):
    confidence = pred_probs[pred_label] * 100
    match = "✓" if pred_label == true_label else "✗"
    print(f"Sample {idx+1}: True={true_label}, Predicted={pred_label}, "
          f"Confidence={confidence:.2f}% {match}")

# ==================== TRAINING HISTORY VISUALIZATION ====================
print("\n" + "="*60)
print("CREATING TRAINING HISTORY PLOTS")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.axhline(y=0.95, color='r', linestyle='--', label='95% Target', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ Training history saved as 'training_history.png'")

# ==================== CONFUSION MATRIX ====================
print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX")
print("="*60)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Predict all test samples
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - MNIST CNN', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# Print classification report
print("\nClassification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# ==================== SAVE MODEL ====================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model.save('mnist_cnn_model.h5')
print("✓ Model saved as 'mnist_cnn_model.h5'")

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Model Architecture: CNN with {model.count_params():,} parameters")
print(f"✓ Training Epochs: {len(history.history['accuracy'])}")
print(f"✓ Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"✓ Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Target Achieved: {'YES' if test_accuracy > 0.95 else 'NO'}")
print(f"\nGenerated Files:")
print(f"  - mnist_cnn_model.h5 (trained model)")
print(f"  - mnist_predictions.png (predictions on 5 samples)")
print(f"  - training_history.png (accuracy and loss curves)")
print(f"  - confusion_matrix.png (classification confusion matrix)")
print("="*60)
