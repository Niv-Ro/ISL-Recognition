"""
Hebrew Sign Language Recognition Application - GUI (Gesture Controlled)

This script launches a Tkinter-based graphical user interface for real-time
Hebrew Sign Language recognition using a pre-trained deep learning model.
It captures video from a webcam, processes hand gestures, predicts the corresponding
Hebrew letter, and allows the user to construct text using head and hand gestures.

Features:
- Real-time hand tracking and sign prediction with a single hand.
- Gesture controls for all text operations:
  - Two Hands: Add the predicted letter.
  - Head Nod: Add a space or convert to a final letter.
  - Head Shake: Delete the last character (backspace).
  - Hand Close to Camera: Clear all text.
- Display of predicted letter and confidence score.
- Text construction area that is manipulated entirely by gestures.
- FPS counter for performance monitoring.
- Graceful handling of model loading and camera operations.
- Detailed preprocessing steps to match dataset creation conditions for optimal accuracy.
- Stability delay to prevent incorrect predictions of transitional movements.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector # type: ignore
import pickle
import os
import threading
import time
from collections import deque

# --- Configuration and Constants ---
# These constants define filenames, hand detection parameters, drawing styles,
# and mappings for Hebrew final letters. They are crucial for consistent operation
# and matching dataset creation conditions.

MODEL_FILENAME = 'hebrew_sign_language_model_224_optimized.keras' # Filename for the trained Keras model.
METADATA_FILENAME = 'model_metadata_224_optimized.pkl' # Filename for the model's metadata (e.g., label mappings).

# --- Parameters to match the dataset creation environment ---
DATASET_DETECTION_CONFIDENCE = 0.5  # Hand detection confidence threshold used when creating the dataset.
DATASET_TRACKING_CONFIDENCE = 0.5  # Hand tracking confidence threshold used when creating the dataset.
DATASET_MODEL_COMPLEXITY = 1      # Model complexity for the hand detector (0, 1, or 2). Higher is more accurate but slower.
DATASET_OFFSET = 20               # Pixel offset around the hand's bounding box to ensure the whole hand is captured.
DATASET_IMG_SIZE = 300            # The reference image size the model was trained on.

# --- Gesture and Prediction Timing Controls ---
STABILITY_DELAY = 0.4 # Time in seconds a single hand must be present and still before making a prediction. Prevents recognizing transitional movements.
TWO_HANDS_STABILITY_DELAY = 0.3 # Time in seconds two hands must be stable before triggering the 'add letter' action.
HEAD_GESTURE_DURATION = 1.0 # Duration in seconds a head movement (nod/shake) must be sustained to be considered a valid gesture.

# --- Visual Styling for Model Input ---
# These styles are used to draw on the image *before* it's fed to the model,
# exactly replicating the conditions under which the training data was created.
LANDMARK_DRAWING_SPEC = {
    'thickness': 6,
    'circle_radius': 10,
    'color': (0, 0, 255) # Red landmarks
}
CONNECTION_DRAWING_SPEC = {
    'thickness': 2,
    'color': (255, 255, 255) # White connections
}

# Mapping for converting regular Hebrew letters to their final (sofit) form, used when adding a space.
FINAL_LETTERS_MAP = {
    'צ': 'ץ',
    'מ': 'ם',
    'נ': 'ן',
    'כ': 'ך',
    'פ': 'ף'
}


class HebrewSignLanguageApp:
    """
    Main application class for the Hebrew Sign Language Recognition GUI.

    Manages the UI, camera feed, hand detection, model prediction,
    and user interactions, ensuring preprocessing matches dataset creation.
    """

    def __init__(self, root_window: tk.Tk):
        """
        Initializes the application.

        Args:
            root_window (tk.Tk): The main Tkinter window.
        """
        # --- Main Window Setup ---
        self.root = root_window
        self.root.title("זיהוי שפת הסימנים הישראלית") # "Israeli Sign Language Recognition"
        self.root.geometry("1200x850")
        self.root.configure(bg='#f0f0f0')

        # --- Core Components Initialization ---
        self.cap = None # Will hold the OpenCV video capture object.
        # Initialize the hand detector to detect up to two hands for gesture control.
        self.detector = HandDetector(
            staticMode=False,
            maxHands=2, # We need to detect two hands for the "add letter" gesture.
            modelComplexity=DATASET_MODEL_COMPLEXITY,
            detectionCon=DATASET_DETECTION_CONFIDENCE,
            minTrackCon=DATASET_TRACKING_CONFIDENCE
        )
        self.model: tf.keras.Model | None = None # Will hold the loaded TensorFlow model.
        self.metadata: dict | None = None # Will hold the loaded model metadata.
        # Load the pre-trained Haar Cascade for frontal face detection.
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            self.face_cascade = None # If it fails, face detection will be disabled.
            print(f"Warning: Could not load face cascade classifier: {e}")

        # --- State Management Variables ---
        self.is_camera_on: bool = False # Flag to control the main camera loop.
        self.current_text: str = "" # Stores the string of recognized letters.
        self.last_predicted_letter: str = "" # Stores the most recent stable prediction.
        self.last_predicted_confidence: float = 0.0 # Stores the confidence of the last prediction.
        self.last_prediction_time: float = 0 # Timestamp of the last prediction (not currently used, but good for future features).
        self.prediction_interval: float = 0.2 # Interval between predictions (not currently used).

        # --- Gesture Control State ---
        self.last_action_time: float = 0 # Timestamp of the last gesture action to implement a cooldown.
        self.action_cooldown: float = 1.2 # Cooldown in seconds to prevent accidental, rapid-fire gesture triggers.
        self.two_hands_present: bool = False # Flag indicating if two hands are currently detected.
        self.is_hand_too_close: bool = False # Flag for the "clear all" gesture.
        # Deques to store recent head positions for shake/nod detection.
        self.head_x_history = deque(maxlen=10)
        self.head_y_history = deque(maxlen=10)
        self.is_head_shaking: bool = False # Flag indicating a head shake gesture is active.
        self.is_head_nodding: bool = False # Flag indicating a head nod gesture is active.

        # --- Head Gesture Timing State ---
        # These states help ensure a head movement is intentional and sustained.
        self.nod_start_time: float = 0 # Timestamp when a potential nod begins.
        self.is_nodding_potential: bool = False # True if a nod-like movement is detected.
        self.shake_start_time: float = 0 # Timestamp when a potential shake begins.
        self.is_shaking_potential: bool = False # True if a shake-like movement is detected.

        # --- Hand Stability Control State ---
        # These states prevent predictions/actions while hands are entering/leaving the frame.
        self.hand_entry_time: float = 0 # Timestamp when a single hand appears.
        self.is_hand_stable: bool = False # Becomes true after the single hand is stable for STABILITY_DELAY.
        self.last_hand_count: int = 0 # The number of hands detected in the previous frame.
        self.two_hands_entry_time: float = 0 # Timestamp when two hands appear.
        self.are_two_hands_stable: bool = False # Becomes true after two hands are stable for TWO_HANDS_STABILITY_DELAY.

        # --- Prediction Smoothing Parameters ---
        self.prediction_history: list = [] # Stores the last few predictions to smooth out results.
        self.prediction_history_size: int = 5 # Number of predictions to keep in the history.
        self.confidence_threshold: float = 0.6 # Minimum confidence for a prediction to be considered valid for smoothing.

        # --- Image Processing Parameters (initialized from constants) ---
        self.offset: int = DATASET_OFFSET
        self.app_img_size: int = DATASET_IMG_SIZE # This may be updated by the loaded metadata.

        # --- Frame Management ---
        self._original_captured_frame_bgr: np.ndarray | None = None # Stores the raw frame from the camera.
        self.display_image_aspect_ratio: float = 640 / 480 # Default aspect ratio, updated on first frame.

        # --- FPS Calculation ---
        self._last_frame_time: float = time.time() # Timestamp of the last frame processed for FPS calculation.
        self._frame_count: int = 0 # Counter for frames within a one-second interval.

        # --- UI and App Initialization ---
        self.setup_ui() # Build the graphical user interface.
        self.load_trained_model() # Load the ML model and metadata.

        # Schedule the camera to start shortly after the UI is ready.
        self.root.after(500, self.start_camera)
        # Define a custom closing protocol to shut down resources gracefully.
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self) -> None:
        """
        Creates and arranges all the graphical user interface (GUI) elements
        for the application, including frames, labels, buttons, and text areas.
        """
        # Create a frame for the main title bar.
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=10, pady=(10, 5))
        title_frame.pack_propagate(False) # Prevent the frame from shrinking to fit its content.
        title_label = tk.Label(title_frame, text="🤟 זיהוי שפת הסימנים הישראלית 🤟",
                               font=('Arial', 22, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)

        # Main frame that holds the two primary panels (camera and text).
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        # Configure grid layout: row 0 takes all available vertical space.
        main_frame.grid_rowconfigure(0, weight=1)
        # The camera panel (column 0) gets 4x more width than the text panel (column 1).
        main_frame.grid_columnconfigure(0, weight=4)
        main_frame.grid_columnconfigure(1, weight=1)

        # --- Left Panel: Camera Feed and Status ---
        left_panel = tk.Frame(main_frame, bg='#f0f0f0')
        left_panel.grid(row=0, column=0, padx=(0, 5), sticky="nsew")
        left_panel.grid_rowconfigure(0, weight=1) # The camera frame within this panel should expand.
        left_panel.grid_columnconfigure(0, weight=1)

        # A labeled frame to contain the camera feed.
        camera_frame = tk.LabelFrame(left_panel, text="📷 מצלמה", # "Camera"
                                     font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50',
                                     relief='raised', bd=2)
        camera_frame.grid(row=0, column=0, pady=(0, 5), sticky="nsew")
        camera_frame.grid_rowconfigure(0, weight=1) # Make the inner grid responsive.
        camera_frame.grid_columnconfigure(0, weight=1)

        # The label that will display the actual camera video feed.
        self.camera_label = tk.Label(camera_frame, text="Camera Starting...", font=('Arial', 16), bg='#34495e', fg='white')
        self.camera_label.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")

        # Container for status, prediction, and FPS labels below the camera.
        controls_container_frame = tk.Frame(left_panel, bg='#f0f0f0')
        controls_container_frame.grid(row=1, column=0, sticky="nsew")
        controls_container_frame.grid_columnconfigure(0, weight=1)

        # A frame to group the status and prediction information.
        status_pred_frame = tk.Frame(controls_container_frame, bg='white', relief='sunken', bd=1, padx=10, pady=5)
        status_pred_frame.grid(row=0, column=0, pady=5, sticky="ew")
        status_pred_frame.grid_columnconfigure(0, weight=1)

        # Label to show the application's current status (e.g., loading model, camera on).
        self.status_label = tk.Label(status_pred_frame, text="סטטוס: טוען...", # "Status: Loading..."
                                     font=('Arial', 12), bg='#ecf0f1', fg='#2c3e50', relief='sunken', bd=1)
        self.status_label.grid(row=0, column=0, columnspan=2, pady=5, padx=5, sticky="ew")

        # A frame to hold the predicted letter and its confidence score.
        pred_label_frame = tk.Frame(status_pred_frame, bg='white', relief='ridge', bd=1, padx=5, pady=5)
        pred_label_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=5, sticky="ew")
        pred_label_frame.grid_columnconfigure(0, weight=1)
        pred_label_frame.grid_columnconfigure(1, weight=1)

        # Label for the predicted letter.
        self.prediction_label = tk.Label(pred_label_frame, text="אות: -", # "Letter: -"
                                         font=('Arial', 16, 'bold'), bg='white', fg='#27ae60')
        self.prediction_label.grid(row=0, column=0, sticky="w", padx=5)

        # Label for the prediction confidence.
        self.confidence_label = tk.Label(pred_label_frame, text="אמינות: 0%", # "Confidence: 0%"
                                         font=('Arial', 12), bg='white', fg='#7f8c8d')
        self.confidence_label.grid(row=0, column=1, sticky="e", padx=5)

        # Label for displaying the frames per second (FPS) of the camera loop.
        self.fps_label = ttk.Label(controls_container_frame, text="FPS: 0.0", font=('Arial', 10), foreground='#555')
        self.fps_label.grid(row=3, column=0, sticky="w", padx=10, pady=(5, 0))

        # --- Right Panel: Text Display (No buttons) ---
        text_panel_frame = tk.LabelFrame(main_frame, text="📝 הטקסט שלך", # "Your Text"
                                         font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50',
                                         relief='raised', bd=2)
        text_panel_frame.grid(row=0, column=1, padx=(5, 0), sticky="nsew")
        text_panel_frame.grid_rowconfigure(0, weight=1)
        text_panel_frame.grid_columnconfigure(0, weight=1)

        # ScrolledText widget to display the composed text.
        self.text_display = scrolledtext.ScrolledText(text_panel_frame, font=('Arial', 20),
                                                      bg='#f8f9fa', fg='#2c3e50', height=10, width=25,
                                                      relief='sunken', bd=2, wrap='word')
        self.text_display.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
        self.update_text_display() # Initialize with placeholder text.

        # --- Bottom Information Bar for Gesture Controls ---
        info_bar_frame = tk.Frame(self.root, bg='#34495e', height=50)
        info_bar_frame.pack(fill='x', padx=10, pady=(5, 10))
        info_bar_frame.pack_propagate(False)
        info_text = "הוספת אות: הכנס יד שנייה לפריים  |  רווח: הנהון ראש  |  מחיקה: נענוע ראש  |  ניקוי הכל: קרב יד למצלמה"
        # "Add Letter: Bring second hand to frame | Space: Nod head | Delete: Shake head | Clear All: Bring hand close to camera"
        info_bar_label = tk.Label(info_bar_frame, text=info_text,
                                  font=('Arial', 16, 'bold'), fg='white', bg='#34495e')
        info_bar_label.pack(expand=True)

    def load_trained_model(self) -> None:
        """Loads the model and metadata, handling errors."""
        try:
            # Check if necessary files exist before attempting to load.
            if not (os.path.exists(MODEL_FILENAME) and os.path.exists(METADATA_FILENAME)):
                raise FileNotFoundError("Model or metadata files not found.")

            # Load the serialized Keras model.
            self.model = tf.keras.models.load_model(MODEL_FILENAME)

            # Load the pickled metadata file.
            with open(METADATA_FILENAME, 'rb') as f:
                self.metadata = pickle.load(f)

            # If the metadata contains the image size, use it to override the default.
            if self.metadata and 'img_size' in self.metadata:
                self.app_img_size = self.metadata['img_size']

            # Update status label to show success.
            self.status_label.config(text="סטטוס: המודל נטען בהצלחה 👍", bg='#d4edda', fg='#155724') # "Status: Model loaded successfully"
            print("✅ Model and metadata loaded successfully!")
        except Exception as e:
            # If loading fails, show a critical error and terminate the application.
            messagebox.showerror("שגיאה קריטית", f"שגיאה קריטית בטעינת המודל: {e}\n\nהאפליקציה תיסגר.") # "Critical Error"
            self.root.destroy()

    def start_camera(self) -> None:
        """Initializes and starts the camera feed."""
        if self.is_camera_on: return # Don't start if it's already running.

        try:
            # Try to open the external camera first (index 1).
            self.cap = cv2.VideoCapture(1)
            # If it fails, fall back to the default/internal camera (index 0).
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            # If both fail, raise an error.
            if not self.cap.isOpened():
                raise ConnectionError("Cannot open camera.")

            # Read a single frame to get its dimensions.
            ret, frame = self.cap.read()
            if ret and frame is not None:
                h, w, _ = frame.shape
                # Set the aspect ratio for proper resizing later.
                self.display_image_aspect_ratio = w / h
                self._original_captured_frame_bgr = frame.copy()

            self.is_camera_on = True
            # Update status label.
            self.status_label.config(text="סטטוס: המצלמה פועלת 🟢", bg='#d4edda', fg='#155724') # "Status: Camera is active"

            # Run the camera loop in a separate thread to prevent the GUI from freezing.
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
        except Exception as e:
            # Show an error message if the camera cannot be started.
            messagebox.showerror("Camera Start Error", f"Could not start camera:\n{str(e)}")
            self.status_label.config(text="סטטוס: שגיאה בהפעלת המצלמה ❌", bg='#f8d7da', fg='#721c24') # "Status: Error starting camera"

    def camera_loop(self) -> None:
        """Main loop for camera, gesture detection, and prediction."""
        last_fps_time = time.time()
        frame_count = 0

        while self.is_camera_on:
            try:
                # Read a frame from the camera.
                success, img_bgr = self.cap.read()
                if not success or img_bgr is None:
                    time.sleep(0.05) # Wait briefly if a frame is missed.
                    continue

                # Create copies for processing and display to avoid modifying the original.
                self._original_captured_frame_bgr = img_bgr.copy()
                img_to_display = img_bgr.copy()
                current_time = time.time()

                # --- Gesture and State Detection ---
                # Find hands in the current frame but don't draw them yet.
                hands, _ = self.detector.findHands(img_to_display, draw=False, flipType=False)

                # Manually draw landmarks to match the dataset style.
                if hands:
                    for hand in hands:
                        if 'lmList' in hand and hand['lmList']:
                            # This drawing is for display purposes only.
                            img_to_display = self.draw_landmarks_exact_match(img_to_display, hand['lmList'])

                # Detect faces for head gesture tracking.
                face_rects = self.detect_face(img_bgr)
                # Check for sustained head movements.
                head_shake_detected = self.detect_head_shake(face_rects, current_time)
                head_nod_detected = self.detect_head_nod(face_rects, current_time)
                hand_count = len(hands)

                # --- Single-Hand Stability Control ---
                # If a single hand has just entered the frame...
                if hand_count == 1 and self.last_hand_count != 1:
                    self.hand_entry_time = current_time # Start the stability timer.
                    self.is_hand_stable = False
                # If a single hand is present but not yet stable...
                elif hand_count == 1 and not self.is_hand_stable:
                    # Check if enough time has passed to consider it stable.
                    if (current_time - self.hand_entry_time) > STABILITY_DELAY:
                        self.is_hand_stable = True
                # If there's no longer one hand, reset stability.
                elif hand_count != 1:
                    self.is_hand_stable = False

                # --- Two-Hand Stability Control (similar logic as single hand) ---
                if hand_count == 2 and self.last_hand_count != 2:
                    self.two_hands_entry_time = current_time
                    self.are_two_hands_stable = False
                elif hand_count == 2 and not self.are_two_hands_stable:
                    if (current_time - self.two_hands_entry_time) > TWO_HANDS_STABILITY_DELAY:
                        self.are_two_hands_stable = True
                elif hand_count != 2:
                    self.are_two_hands_stable = False

                # --- Action Logic ---
                action_taken = False
                # Only process gestures if the cooldown period has passed.
                if current_time - self.last_action_time > self.action_cooldown:
                    # Gesture 1: Clear text if a hand is very close to the camera (takes up > 40% of view area).
                    if hand_count > 0 and (hands[0]['bbox'][2] * hands[0]['bbox'][3]) > (img_bgr.shape[0] * img_bgr.shape[1] * 0.4):
                        if not self.is_hand_too_close: # Trigger only on the first frame it's detected.
                            self.root.after(0, self.trigger_clear_text_action)
                            action_taken = True
                    # Gesture 2: Add letter if two hands are stable.
                    elif self.are_two_hands_stable and not self.two_hands_present:
                        letter_to_add = self.last_predicted_letter
                        # Use root.after to ensure UI updates happen on the main thread.
                        self.root.after(0, lambda l=letter_to_add: self.trigger_add_letter_action(l))
                        action_taken = True
                    # Gesture 3: Add space/final letter on head nod.
                    elif head_nod_detected and not self.is_head_nodding:
                        self.root.after(0, self.trigger_space_action)
                        action_taken = True
                    # Gesture 4: Backspace on head shake.
                    elif head_shake_detected and not self.is_head_shaking:
                        self.root.after(0, self.trigger_backspace_action)
                        action_taken = True

                    # If any action was taken, reset the cooldown timer.
                    if action_taken:
                        self.last_action_time = current_time

                # --- Update Gesture States for the Next Frame ---
                self.two_hands_present = self.are_two_hands_stable
                self.is_head_shaking = head_shake_detected
                self.is_head_nodding = head_nod_detected
                # Recalculate if hand is too close for the next frame.
                self.is_hand_too_close = hand_count > 0 and (hands[0]['bbox'][2] * hands[0]['bbox'][3]) > (img_bgr.shape[0] * img_bgr.shape[1] * 0.4)
                self.last_hand_count = hand_count

                # --- Prediction Logic ---
                # Only predict if there's a single, stable hand that isn't too close.
                should_predict = self.is_hand_stable and not self.is_hand_too_close
                if should_predict:
                    # Process the frame to match the model's input requirements.
                    img_for_model = self.process_frame_for_prediction(img_bgr, hands[0])
                    if img_for_model is not None:
                        # Perform the prediction.
                        letter, confidence = self.predict_sign(img_for_model)
                        # Schedule the UI update on the main thread.
                        self.root.after(0, lambda l=letter, c=confidence: self.update_prediction_display(l, c))
                # If there are no hands, clear the prediction display.
                elif hand_count == 0:
                    self.root.after(0, self.update_prediction_display, "-", 0.0)

                # --- Final Display Updates ---
                self.draw_all_feedback(img_to_display) # Overlay text feedback for gestures.
                self.update_camera_display(img_to_display) # Render the final image to the GUI.

                # --- FPS Calculation ---
                frame_count += 1
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}")) # Update FPS label.
                    frame_count = 0
                    last_fps_time = current_time

            except Exception as e:
                # Print errors from the camera loop for debugging, but don't crash.
                print(f"ERROR IN CAMERA LOOP: {e}")
                time.sleep(0.1)

    def process_frame_for_prediction(self, img_bgr_original: np.ndarray, hand: dict) -> np.ndarray | None:
        """Processes the frame to create the input for the model, with exact landmark drawing."""
        img_for_model_processing = img_bgr_original.copy()
        x, y, w, h = hand['bbox']

        if 'lmList' in hand and hand['lmList']:
            try:
                # CRITICAL STEP: Draw landmarks and connections onto the image itself.
                # The model was trained on images with these drawings, so it expects them for inference.
                img_for_model_processing = self.draw_landmarks_exact_match(img_for_model_processing, hand['lmList'])
                # Now crop, pad, and resize the hand from this modified image.
                return self.process_hand_image_for_prediction(img_for_model_processing, x, y, w, h)
            except Exception as e:
                print(f"Warning: Landmark drawing or processing failed. Error: {e}")
                return None
        return None

    def process_hand_image_for_prediction(self, img_full_frame_with_landmarks: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray | None:
        """Crops, pads, and resizes the hand image to match model input requirements."""
        try:
            # Apply offset to the bounding box to ensure the entire hand is included.
            x1 = max(x - self.offset, 0)
            y1 = max(y - self.offset, 0)
            x2 = min(x + w + self.offset, img_full_frame_with_landmarks.shape[1])
            y2 = min(y + h + self.offset, img_full_frame_with_landmarks.shape[0])

            # Crop the hand region.
            img_crop = img_full_frame_with_landmarks[y1:y2, x1:x2]

            # If crop is invalid, abort.
            if img_crop.size == 0: return None

            # Create a square white canvas to place the cropped hand on.
            target_s = self.app_img_size
            img_white_canvas = np.ones((target_s, target_s, 3), np.uint8) * 255

            # --- Resizing logic to maintain aspect ratio ---
            crop_h, crop_w = img_crop.shape[:2]
            aspect_ratio = crop_h / crop_w

            if aspect_ratio > 1: # If height is greater than width
                scale_factor = target_s / crop_h
                new_w = int(crop_w * scale_factor)
                img_resized = cv2.resize(img_crop, (new_w, target_s))
                # Center the resized image horizontally.
                w_gap = (target_s - new_w) // 2
                img_white_canvas[:, w_gap:w_gap + new_w] = img_resized
            else: # If width is greater than or equal to height
                scale_factor = target_s / crop_w
                new_h = int(crop_h * scale_factor)
                img_resized = cv2.resize(img_crop, (target_s, new_h))
                # Center the resized image vertically.
                h_gap = (target_s - new_h) // 2
                img_white_canvas[h_gap:h_gap + new_h, :] = img_resized

            return img_white_canvas
        except Exception as e:
            print(f"Error in process_hand_image_for_prediction: {e}")
            return None

    def predict_sign(self, img_processed: np.ndarray) -> tuple[str, float]:
        """Performs prediction on a processed image and manages history."""
        if self.model is None or self.metadata is None: return "?", 0.0 # Safety check.
        try:
            # Prepare the image for the model: BGR -> RGB, normalize to [0,1], add batch dimension.
            img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_norm, axis=0)

            # Get model's prediction logits.
            prediction = self.model.predict(img_batch, verbose=0)[0]

            # Find the index with the highest probability.
            idx = np.argmax(prediction)
            confidence = float(np.max(prediction))
            # Map the index back to a letter label using the loaded metadata.
            letter = self.metadata.get('index_to_label', {}).get(idx, "?")

            # Add the raw prediction to our history for smoothing.
            self.prediction_history.append((letter, confidence))
            if len(self.prediction_history) > self.prediction_history_size:
                self.prediction_history.pop(0)

            # Return the smoothed prediction, not the raw one.
            return self.get_smoothed_prediction()
        except Exception as e:
            print(f"Error in predict_sign: {e}")
            return "?", 0.0

    def get_smoothed_prediction(self) -> tuple[str, float]:
        """
        Analyzes the `self.prediction_history` to provide a more stable prediction.
        It filters predictions by `self.confidence_threshold` and then returns the
        most frequent letter among these confident predictions, along with its
        average confidence. If no predictions meet the threshold, it returns
        the most recent raw prediction.
        """
        if not self.prediction_history:
            return "?", 0.0

        # Filter out predictions that are below the confidence threshold.
        high_confidence_predictions = [
            (letter, conf) for letter, conf in self.prediction_history
            if conf >= self.confidence_threshold
        ]

        # If no recent predictions were confident, return the latest one regardless of confidence.
        if not high_confidence_predictions:
            return self.prediction_history[-1]

        # Count the occurrences of each letter in the confident predictions.
        letter_counts: dict[str, int] = {}
        letter_confidences: dict[str, list[float]] = {}
        for letter, conf in high_confidence_predictions:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            if letter not in letter_confidences:
                letter_confidences[letter] = []
            letter_confidences[letter].append(conf)

        if not letter_counts:
             return self.prediction_history[-1] # Fallback just in case.

        # Find the letter that appeared most frequently.
        most_frequent_letter = max(letter_counts, key=lambda k_val: letter_counts[k_val])
        # Calculate the average confidence for that specific letter.
        avg_confidence_for_most_frequent = sum(letter_confidences[most_frequent_letter]) / len(letter_confidences[most_frequent_letter])

        return most_frequent_letter, avg_confidence_for_most_frequent

    # --- Gesture Detection Helper Methods ---
    def detect_face(self, img_bgr: np.ndarray) -> list:
        """Detects faces in the frame using the Haar Cascade."""
        if self.face_cascade is None: return [] # Skip if cascade isn't loaded.
        # Convert to grayscale for the cascade classifier.
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Perform detection.
        return self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    def detect_head_shake(self, faces, current_time: float) -> bool:
        """Detects a side-to-side head shake gesture."""
        if len(faces) == 0:
            # If no face is detected, reset the history and state.
            self.head_x_history.clear()
            self.is_shaking_potential = False
            return False

        # Get the horizontal center of the detected face.
        center_x = faces[0][0] + faces[0][2] // 2
        self.head_x_history.append(center_x)
        if len(self.head_x_history) < 10: return False # Need enough data points.

        # Check if the horizontal movement range exceeds a threshold.
        if (max(self.head_x_history) - min(self.head_x_history)) > 40: # 40 pixel threshold
            # If this is the start of a potential shake, record the time.
            if not self.is_shaking_potential:
                self.is_shaking_potential = True
                self.shake_start_time = current_time

            # If the movement has been sustained for the required duration, it's a valid gesture.
            if self.is_shaking_potential and (current_time - self.shake_start_time > HEAD_GESTURE_DURATION):
                self.is_shaking_potential = False # Reset for the next gesture.
                self.head_x_history.clear()
                return True
        else:
            # If movement is below the threshold, it's not a shake.
            self.is_shaking_potential = False

        return False

    def detect_head_nod(self, faces, current_time: float) -> bool:
        """Detects an up-and-down head nod gesture."""
        if len(faces) == 0:
            # If no face is detected, reset history and state.
            self.head_y_history.clear()
            self.is_nodding_potential = False
            return False

        # Get the vertical center of the detected face.
        center_y = faces[0][1] + faces[0][3] // 2
        self.head_y_history.append(center_y)
        if len(self.head_y_history) < 10: return False # Need enough data points.

        # Check if the vertical movement range exceeds a threshold.
        if (max(self.head_y_history) - min(self.head_y_history)) > 35: # 35 pixel threshold
            # If this is the start of a potential nod, record the time.
            if not self.is_nodding_potential:
                self.is_nodding_potential = True
                self.nod_start_time = current_time

            # If the movement has been sustained for the required duration, it's a valid gesture.
            if self.is_nodding_potential and (current_time - self.nod_start_time > HEAD_GESTURE_DURATION):
                self.is_nodding_potential = False # Reset for the next gesture.
                self.head_y_history.clear()
                return True
        else:
            # If movement is below the threshold, it's not a nod.
            self.is_nodding_potential = False

        return False

    # --- UI Update and Action Trigger Methods ---
    def trigger_add_letter_action(self, letter_to_add: str):
        """Adds the last predicted letter to the text display."""
        print("ACTION: Two Hands -> Add letter")
        # Only add a valid, recognized letter.
        if letter_to_add and letter_to_add != "?":
            # If the display has the placeholder text, clear it first.
            if self.text_display.get("1.0", tk.END).strip() == "הטקסט שלך יופיע כאן...":
                self.current_text = ""
            self.current_text += letter_to_add
            self.update_text_display()
            # Clear history and prediction to prevent accidental re-addition.
            self.prediction_history.clear()
            self.update_prediction_display("-", 0.0)

    def trigger_space_action(self):
        """Adds a space or converts the last letter to its final form."""
        print("ACTION: Head Nod -> Add space")
        # Ignore if text area is empty or has placeholder.
        if not self.current_text or self.text_display.get("1.0", tk.END).strip() == "הטקסט שלך יופיע כאן...":
            return
        # Check if the last character has a final form (sofit).
        if self.current_text and self.current_text[-1] in FINAL_LETTERS_MAP:
            # Replace it with the final form.
            self.current_text = self.current_text[:-1] + FINAL_LETTERS_MAP[self.current_text[-1]]
        # Add a space after the letter (or the converted final letter).
        self.current_text += " "
        self.update_text_display()

    def trigger_backspace_action(self):
        """Deletes the last character from the text display."""
        print("ACTION: Head Shake -> Backspace")
        if self.current_text and self.text_display.get("1.0", tk.END).strip() != "הטקסט שלך יופיע כאן...":
            # Simple string slicing to remove the last character.
            self.current_text = self.current_text[:-1]
            self.update_text_display()

    def trigger_clear_text_action(self):
        """Clears all text from the display."""
        print("ACTION: Hand Close -> Clear Text")
        self.current_text = ""
        self.prediction_history.clear()
        self.update_prediction_display("-", 0.0)
        self.update_text_display()

    def draw_all_feedback(self, image: np.ndarray):
        """Draws visual feedback for gestures on the display image (TEXT ONLY)."""
        # This provides immediate visual confirmation of a detected gesture.
        if self.two_hands_present:
            cv2.putText(image, "ADD LETTER", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
        if self.is_head_nodding:
            cv2.putText(image, "SPACE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 200, 255), 5, cv2.LINE_AA)
        if self.is_head_shaking:
            cv2.putText(image, "DELETE", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 5, cv2.LINE_AA)
        if self.is_hand_too_close:
            cv2.putText(image, "CLEAR ALL", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

    def update_camera_display(self, img_bgr: np.ndarray):
        """Updates the camera label with a new frame."""
        try:
            # Convert OpenCV BGR image to RGB for PIL.
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Get the current size of the Tkinter label.
            label_w, label_h = self.camera_label.winfo_width(), self.camera_label.winfo_height()

            # Resize the image to fit the label while maintaining aspect ratio.
            if label_w > 1 and label_h > 1:
                img_pil.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)

            # Convert PIL image to a Tkinter-compatible format.
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update the label's image.
            self.camera_label.config(image=img_tk, text="")
            # Keep a reference to the image to prevent it from being garbage collected.
            self.camera_label.image = img_tk
        except (RuntimeError, tk.TclError):
            # Ignore errors that can happen when the UI is being torn down.
            pass

    def update_text_display(self) -> None:
        """Updates the text display widget."""
        self.text_display.config(state='normal') # Enable editing.
        self.text_display.delete('1.0', tk.END) # Clear existing content.
        # Insert the current text, or the placeholder if the text is empty.
        self.text_display.insert('1.0', self.current_text if self.current_text.strip() else "הטקסט שלך יופיע כאן...") # "Your text will appear here..."
        self.text_display.config(state='disabled') # Disable editing to make it read-only for the user.

    def update_prediction_display(self, letter: str, confidence: float) -> None:
        """Updates the prediction and confidence labels."""
        self.last_predicted_letter = letter

        # Change color and text based on confidence level for better user feedback.
        if confidence >= 0.8: c, t = '#27ae60', "גבוהה מאוד" # Very High
        elif confidence >= 0.65: c, t = '#2980b9', "טובה" # Good
        elif confidence >= 0.45: c, t = '#f39c12', "בינונית" # Medium
        else: c, t = '#e74c3c', "נמוכה" # Low

        # Update the labels with the new information.
        self.prediction_label.config(text=f"אות: {letter}", fg=c) # "Letter:"
        self.confidence_label.config(text=f"אמינות: {confidence:.1%} ({t})") # "Confidence:"

    def on_camera_label_resize(self, event: tk.Event) -> None:
        """Placeholder for handling camera label resize events."""
        # This function was likely used for more complex responsive logic previously.
        # It's kept here in case it's needed again.
        pass

    def on_closing(self) -> None:
        """Handles window closing event."""
        print("INFO: Closing application...")
        # Signal the camera loop to stop.
        self.is_camera_on = False
        # Wait briefly for the camera thread to finish.
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=0.5)
        # Release the camera resource.
        if self.cap: self.cap.release()
        # Destroy the Tkinter window.
        self.root.destroy()

    def draw_landmarks_exact_match(self, img: np.ndarray, hand_landmarks_list: list) -> np.ndarray:
        """Draws landmarks and connections to EXACTLY match dataset creation style."""
        try:
            # This method ensures the pre-processing precisely matches the dataset images.
            import mediapipe as mp
            mp_hands = mp.solutions.hands

            # Draw connections between landmarks.
            for conn in mp_hands.HAND_CONNECTIONS:
                start, end = conn
                if start < len(hand_landmarks_list) and end < len(hand_landmarks_list):
                    start_pt = (hand_landmarks_list[start][0], hand_landmarks_list[start][1])
                    end_pt = (hand_landmarks_list[end][0], hand_landmarks_list[end][1])
                    cv2.line(img, start_pt, end_pt, CONNECTION_DRAWING_SPEC['color'], CONNECTION_DRAWING_SPEC['thickness'])

            # Draw the landmarks themselves as circles.
            for lm in hand_landmarks_list:
                cv2.circle(img, (lm[0], lm[1]), LANDMARK_DRAWING_SPEC['circle_radius'], LANDMARK_DRAWING_SPEC['color'], -1) # -1 thickness for a filled circle.
        except Exception as e:
            # This might fail if mediapipe is not installed, so print a warning.
            print(f"Warning: Could not draw landmarks with exact match specs: {e}")
        return img


# --- Main Execution Block ---
def main() -> None:
    """Initializes and runs the Tkinter application."""
    # Create the main Tkinter window.
    root = tk.Tk()
    # Instantiate the application class.
    app = HebrewSignLanguageApp(root)
    # Start the Tkinter event loop.
    root.mainloop()

# This ensures the script runs only when executed directly.
if __name__ == "__main__":
    print("🤟 Hebrew Sign Language Recognition App (Gesture Controlled) 🤟")
    main()