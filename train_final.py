"""
This script trains and evaluates a deep learning model for Hebrew Sign Language recognition.
It includes data loading (assuming data is already extracted in DATA_DIR),
a Convolutional Neural Network (CNN) model definition with L2 regularization and Dropout
to mitigate overfitting, data augmentation,
and TensorFlow/Keras callbacks for efficient training.
Class imbalance checking has been removed as the dataset is assumed to be balanced.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.layers import SpatialDropout2D
import matplotlib.pyplot as plt
import pickle
import os

# Check and print the number of available GPUs
print(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

# Define global configurations and constants for the training process
# Configuration and Constants
DATA_DIR = "Data"  # Directory where the data is stored.
IMG_SIZE = 224  # Target image size (height and width in pixels) for model input.
BATCH_SIZE = 32  # Number of samples processed before the model is updated.
EPOCHS = 50  # Number of complete passes through the entire training dataset.
L2_REG_FACTOR = 0.001  # Factor for L2 regularization to penalize large weights.

MODEL_FILENAME = 'hebrew_sign_language_model_224_optimized.keras'  # Filename for saving the trained model.
METADATA_FILENAME = 'model_metadata_224_optimized.pkl'  # Filename for saving model metadata (e.g., class labels).

RETRAIN_MODEL = True # Boolean flag to force retraining even if a saved model exists.

# Define the CNN model architecture
# Model Architecture
def create_model(num_classes, input_shape):
    """
    Creates and compiles a Convolutional Neural Network (CNN) model.

    The model architecture consists of several convolutional blocks followed by
    dense layers. Each convolutional block includes Conv2D layers, Batch Normalization,
    MaxPooling2D, and SpatialDropout2D. L2 regularization is applied to
    convolutional and dense layers.

    Args:
        num_classes (int): The number of output classes (e.g., number of sign language letters).
        input_shape (tuple): The shape of the input images (e.g., (IMG_SIZE, IMG_SIZE, 3)).

    Returns:
        tf.keras.models.Sequential: The compiled Keras model.
    """
    # Start building the sequential model
    model = tf.keras.models.Sequential([
        # First convolutional block with Batch Normalization, MaxPooling, and Spatial Dropout
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape,
                               kernel_regularizer=regularizers.l2(L2_REG_FACTOR)), # First conv layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(L2_REG_FACTOR)), # Second conv layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)), # Pooling layer
        tf.keras.layers.SpatialDropout2D(0.15),  # Spatial Dropout for regularization

        # Second convolutional block
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(L2_REG_FACTOR)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(L2_REG_FACTOR)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(0.2), # Increased dropout

        # Third convolutional block
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(L2_REG_FACTOR)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=regularizers.l2(L2_REG_FACTOR)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(0.25), # Further increased dropout

        # Flatten features for the dense layers
        tf.keras.layers.Flatten(), # Flatten the 3D feature maps to 1D for the dense layers

        # Dense block with Batch Normalization and Dropout
        # Dense Block
        tf.keras.layers.Dense(256, activation='relu', # Dense layer with more units
                              kernel_regularizer=regularizers.l2(L2_REG_FACTOR)),
        tf.keras.layers.BatchNormalization(),  # Batch norm before dropout
        tf.keras.layers.Dropout(0.5), # Standard dropout for dense layers
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax') # Output layer with softmax for classification
    ])

    # Compile the model with optimizer, loss function, and metrics
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), # Adam optimizer with a specific learning rate
                  loss='categorical_crossentropy', # Loss function for multi-class classification
                  metrics=['accuracy']) # Metric to monitor
    return model


# Define the main training and evaluation function
# Training and Evaluation
def train_sign_language_model():
    """
    Trains the sign language recognition model.

    This function performs the following steps:
    1. Sets up data augmentation using ImageDataGenerator.
    2. Loads training and validation datasets from DATA_DIR.
    3. Saves metadata (class labels, image size).
    4. Creates the CNN model.
    5. Defines callbacks for training (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
    6. Trains the model using `model.fit()`.
    7. Evaluates the trained model on the validation set.
    8. Plots and displays the training accuracy and loss history.

    Returns:
        bool: True if training completed successfully, False otherwise.
    """
    print("--- Starting Model Training ---")

    # Set up ImageDataGenerator for data augmentation and preprocessing
    # ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rescale=1. / 255,            # Normalize pixel values to [0, 1]
        rotation_range=8,          # Randomly rotate images by up to 8 degrees
        width_shift_range=0.1,     # Randomly shift images horizontally by up to 10% of width
        height_shift_range=0.1,    # Randomly shift images vertically by up to 10% of height
        shear_range=0.1,           # Apply shear transformations
        zoom_range=[0.9, 1.1],     # Randomly zoom images (90% to 110%)
        horizontal_flip=False,     # No horizontal flipping (can be problematic for signs)
        vertical_flip=False,       # No vertical flipping
        brightness_range=[0.7, 1.3],# Randomly adjust brightness (70% to 130%)
        fill_mode='nearest',       # Strategy for filling in newly created pixels
        validation_split=0.2       # Reserve 20% of data for validation
    )

    # Load training and validation datasets from the specified directory
    # Load training and validation data
    try:
        train_generator = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE), # Resize images
            batch_size=BATCH_SIZE,
            class_mode='categorical', # For multi-class classification
            subset='training',        # Specify this is the training data subset
            color_mode='rgb',         # Images are in color
            shuffle=True)             # Shuffle training data

        validation_generator = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',      # Specify this is the validation data subset
            color_mode='rgb',
            shuffle=False)            # No need to shuffle validation data
    # Handle potential errors during data loading
    except FileNotFoundError: # if DATA_DIR is not found
        print(f" Error: Data directory '{DATA_DIR}' not found. Please ensure it exists and contains the data.")
        return False
    except Exception as e_load: # Catch other potential loading errors
        print(f" Error loading data: {e_load}")
        return False

    # Perform additional checks on the data directory and class discovery
    # Verify data directory again after attempting to load, in case flow_from_directory failed silently or for other reasons
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f" Error: Data directory '{DATA_DIR}' is missing or empty. Please ensure it's populated with class subdirectories.")
        return False

    # Validate class discovery
    if train_generator.num_classes == 0 or validation_generator.num_classes == 0 or \
            train_generator.num_classes != validation_generator.num_classes:
        print(" Error with class discovery or mismatch. Please check DATA_DIR structure and content.")
        print(
            f"Training classes found: {train_generator.num_classes}, Validation classes found: {validation_generator.num_classes}")
        try:
            actual_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
            print(f"Actual subdirectories in '{DATA_DIR}': {actual_dirs} (Count: {len(actual_dirs)})")
            if not actual_dirs: print(f"Warning: '{DATA_DIR}' seems empty or has no subdirectories representing classes.")
        except Exception as e_listdir:
            print(f"Could not list directories in '{DATA_DIR}': {e_listdir}")
        return False

    # Print information about the loaded datasets
    print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
    print(
        f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")

    # Extract class information and prepare metadata
    class_indices = train_generator.class_indices

    hebrew_letters = sorted(list(class_indices.keys())) # Get sorted list of class names
    num_classes = len(hebrew_letters)
    index_to_label = {v: k for k, v in class_indices.items()} # Reverse mapping: index to class name

    # Save the collected metadata to a file
    # Save metadata
    metadata = {'hebrew_letters': hebrew_letters, 'class_indices': class_indices,
                'index_to_label': index_to_label, 'img_size': IMG_SIZE}
    with open(METADATA_FILENAME, 'wb') as f:
        pickle.dump(metadata, f)
    print(f" Metadata saved to '{METADATA_FILENAME}'")

    # Create the CNN model using the defined architecture
    # Create the model
    model = create_model(num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.summary() # Print model architecture

    # Define callbacks for efficient training (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
    # Callbacks for training
    # Stop training if validation loss doesn't improve for 'patience' epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    # Reduce learning rate if validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
    # Save the best model based on validation loss
    checkpoint_cb = ModelCheckpoint(
        MODEL_FILENAME, monitor='val_loss', save_best_only=True,
        save_weights_only=False, # Save the entire model
        mode='min', # Mode for val_loss should be 'min'
        verbose=1
    )

    # Start the model training process
    print(" Starting model training with adjusted settings...")
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, checkpoint_cb]
    )

    # Evaluate the trained model on the validation set
    # Evaluate the model (best weights should be restored by EarlyStopping or loaded by ModelCheckpoint)
    print("\nEvaluating model (best weights should be restored by EarlyStopping or loaded from ModelCheckpoint):")
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Plot and display the training accuracy and loss history
    # Plotting training history
    plt.figure(figsize=(14, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1) # First subplot for accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(1, 2, 2) # Second subplot for loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show() # Display the plots

    return True


# Main execution block: checks conditions and initiates training if needed
if __name__ == "__main__":
    print("Hebrew Sign Language Recognition Model Training ")
    print("=" * 75)

    # Verify that the data directory exists and is not empty before proceeding
    # Verify DATA_DIR exists and is not empty before proceeding
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' is missing, empty, or not accessible.")
        print("Please ensure the 'Data' directory exists in the project root and contains the class subdirectories.")
        # Consider exiting if data is not found, e.g., exit()
    else:
        print(f"Using data from directory: '{DATA_DIR}'")

        # Determine whether to retrain the model or use existing files
        # Check if retraining is forced or if model/metadata don't exist
        if RETRAIN_MODEL or not (os.path.exists(MODEL_FILENAME) and os.path.exists(METADATA_FILENAME)):
            # If retraining is forced, remove any existing model and metadata files
            # If forcing retrain, remove old model and metadata files if they exist
            if RETRAIN_MODEL and os.path.exists(MODEL_FILENAME):
                print(f"RETRAIN_MODEL is True. Removing existing model: {MODEL_FILENAME}")
                os.remove(MODEL_FILENAME)
            if RETRAIN_MODEL and os.path.exists(METADATA_FILENAME):
                print(f"RETRAIN_MODEL is True. Removing existing metadata: {METADATA_FILENAME}")
                os.remove(METADATA_FILENAME)

            # Call the training function
            # Start the training process
            if not train_sign_language_model():
                print("Training process encountered an error or was exited.")
        else:
            # Skip training if model and metadata already exist and retraining is not forced
            # Skip training if model and metadata exist and retraining is not forced
            print(f"Model '{MODEL_FILENAME}' and metadata '{METADATA_FILENAME}' already exist. "
                  f"Skipping training. Set RETRAIN_MODEL=True to force retraining.")

    # Indicate the end of the training script
    print("\n--- Training Script Finished ---")
    # Print paths to the saved model and metadata files for user convenience
    # Print paths to saved files for user convenience
    if os.path.exists(MODEL_FILENAME):
        print(f"Model saved to: {os.path.abspath(MODEL_FILENAME)}")
    if os.path.exists(METADATA_FILENAME):
        print(f"Metadata saved to: {os.path.abspath(METADATA_FILENAME)}")