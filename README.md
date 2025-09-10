# Hebrew Sign Language Recognition 🤟
Real-time Hebrew Sign Language recognition system using computer vision and deep learning - bridging communication gaps for the deaf community.

### 🎯 Impact
Engineered a CNN-based recognition system achieving **91% accuracy** across 22 Hebrew characters, processing **11,000+ samples** to make technology more accessible for the hearing-impaired community.

### ✨ Features
- Real-time gesture recognition for 22 Hebrew alphabet characters
- Custom CNN architecture with optimized hyperparameters
- Data pipeline processing 500 samples per character with augmentation (rotation, scaling, noise injection)
- Gesture-to-text interface with confidence scoring and character prediction visualization
- 25% reduction in training time through hyperparameter tuning (learning rate, batch size, epochs)
- Real-time inference at 30+ FPS using MediaPipe hand landmark detection

### 🛠️ Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![MediaPipe](https://img.shields.io/badge/MediaPipe-00897B?style=for-the-badge&logo=google&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

### 🚀 Quick Start

#### Prerequisites

# Clone the repository
git clone https://github.com/Niv-Ro/ISL-Recognition.git
cd ISL-Recognition

# Install required packages
pip install tensorflow opencv-python mediapipe numpy pandas scikit-learn matplotlib

Step 0: Open Phycharm with the files

Step 1: Data Collection
Collect gesture samples for each Hebrew character:

Run data collection script
python data_collection.py

Instructions:
1. Change the letter destination in the data collection file: folder = "Data/<Character>"
2. run the data collection
3. Position your hand in front of the camera
4. The system will automatically detect your hand using MediaPipe
5. Press the "s" key for each Hebrew letter to start capturing one by one (untill you will have 500)
6. stop running
7. Move to next character (follow step 1)
8. Images are saved in ./data/ directory organized by character

### Step 2: Model Training
Train the CNN model on collected data:

Run training script
python train_final.py

### This will:
- Load 11,000+ samples from ./data/ directory
- Apply data augmentation (rotation, scaling, noise injection)
- Normalize and preprocess images
- Split data 80/20 for training/validation
- Train CNN with optimized hyperparameters
- Implement batch processing for efficient training
- Save trained model to ./model/- Display training metrics and accuracy (91%)

### Step 3: Real-time Recognition
Launch the GUI for real-time gesture recognition:

Run the recognition interface
python gui_ver_2.py

Features:
Real-time hand tracking and sign prediction with a single hand.
Gesture controls for all text operations:
  Two Hands: Add the predicted letter.
  Head Nod: Add a space or convert to a final letter.
  Head Shake: Delete the last character (backspace).
  Hand Close to Camera: Clear all text.
Display of predicted letter and confidence score.
Text construction area that is manipulated entirely by gestures.
FPS counter for performance monitoring.
Graceful handling of model loading and camera operations.
Detailed preprocessing steps to match dataset creation conditions for optimal accuracy.
Stability delay to prevent incorrect predictions of transitional movements.

### video link to project preview
https://youtu.be/zYxUgy_xoqM

