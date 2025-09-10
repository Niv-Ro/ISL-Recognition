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
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![MediaPipe](https://img.shields.io/badge/MediaPipe-00897B?style=for-the-badge&logo=google&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)


### 🎨 Key Features
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

